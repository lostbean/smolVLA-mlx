defmodule SimEnvAdapter.Client do
  @moduledoc """
  The sim env adapter's transport half: a generic `chumak` REQ-socket +
  `msgpax` (MessagePack) client to the Python `sim_server`'s REP socket,
  following the SAME proven pattern as `ControlLoop.ZeroMQClient` (ADR-0007 /
  ADR-0008 / ADR-0012):

    * `msgpax` decodes map keys as strings, never atoms, so replies are
      matched on `"ok"` / `"error"`;
    * a dropped connection is retried by `chumak`'s own peer process once
      `connect/4` has run; what this module guards is the REQ socket's own
      request/reply state machine getting wedged (`:efsm`) or a hung server --
      every round trip is wrapped in a supervised `Task` with a timeout, and a
      timed-out / wedged socket is closed and reopened transparently;
    * fails loud and local: a timeout or connection failure returns
      `{:error, reason}` -- never an indefinite retry, never a swallowed error.

  Where `ControlLoop.ZeroMQClient` pins the wire shape to the infer_action
  observation/action_chunk contract, this client is generic: `request/2`
  sends any MessagePack-encodable map and returns the sim server's `ok`
  payload (a map) or an error, so the sim's `reset` / `step` / `render` ops all
  ride the one round-trip path.
  """

  require Logger

  defstruct [:conn, :host, :port, :timeout_ms]

  @type t :: %__MODULE__{
          conn: pid(),
          host: charlist() | String.t(),
          port: non_neg_integer(),
          timeout_ms: pos_integer()
        }

  @default_timeout_ms 5_000

  @doc """
  Opens a REQ socket connected to the sim server at `host:port`.
  `timeout_ms` bounds every subsequent `request/2` (default
  #{@default_timeout_ms}ms).
  """
  @spec connect(charlist() | String.t(), non_neg_integer(), keyword()) ::
          {:ok, t()} | {:error, term()}
  def connect(host, port, opts \\ []) do
    timeout_ms = Keyword.get(opts, :timeout_ms, @default_timeout_ms)

    case open_socket(host, port) do
      {:ok, socket} ->
        {:ok, conn} = Agent.start_link(fn -> socket end)
        {:ok, %__MODULE__{conn: conn, host: host, port: port, timeout_ms: timeout_ms}}

      {:error, _reason} = error ->
        error
    end
  end

  @doc """
  Sends one request map and returns `{:ok, payload}` (the sim server's `ok`
  payload map) or `{:error, reason}`.

  `reason` is one of:
    * `:timeout` -- the round trip did not complete within `timeout_ms`;
    * `{:server_error, message}` -- the sim server replied with an `error`
      (unknown op, wrong action shape, an env that raised);
    * any other term surfaced by `chumak` (`:no_connected_peers`, `:closed`,
      `:efsm`, ...) or a malformed reply (`{:malformed_response, ...}`).

  Never raises and never blocks past `timeout_ms`.
  """
  @spec request(t(), map()) :: {:ok, map()} | {:error, term()}
  def request(%__MODULE__{} = client, request) when is_map(request) do
    raw_request = Msgpax.pack!(request, iodata: false)
    socket = Agent.get(client.conn, & &1)

    task =
      Task.async(fn ->
        with :ok <- :chumak.send(socket, raw_request),
             {:ok, raw_response} <- :chumak.recv(socket) do
          decode_response(raw_response)
        end
      end)

    case Task.yield(task, client.timeout_ms) || Task.shutdown(task, :brutal_kill) do
      {:ok, {:error, :efsm}} ->
        Logger.warning("SimEnvAdapter.Client: REQ socket wedged (:efsm), reopening")
        reopen(client)
        {:error, :efsm}

      {:ok, result} ->
        result

      nil ->
        Logger.warning("SimEnvAdapter.Client: request timed out after #{client.timeout_ms}ms")
        reopen(client)
        {:error, :timeout}

      {:exit, reason} ->
        {:error, {:task_exit, reason}}
    end
  end

  @doc false
  @spec decode_response(binary()) :: {:ok, map()} | {:error, term()}
  def decode_response(raw_response) do
    case Msgpax.unpack(raw_response) do
      {:ok, %{"ok" => payload}} when is_map(payload) ->
        {:ok, payload}

      {:ok, %{"error" => %{"message" => message}}} ->
        {:error, {:server_error, message}}

      {:ok, other} ->
        {:error, {:malformed_response, other}}

      {:error, reason} ->
        {:error, {:malformed_response, reason}}
    end
  end

  defp open_socket(host, port) do
    with {:ok, socket} <- :chumak.socket(:req),
         {:ok, _peer} <- :chumak.connect(socket, :tcp, to_charlist(host), port) do
      {:ok, socket}
    end
  end

  # Same recovery as ControlLoop.ZeroMQClient: a timed-out / :efsm-wedged call
  # leaves the REQ state machine stuck mid-cycle; closing and reopening the
  # socket resets it, and chumak's own peer retry re-establishes TCP.
  defp reopen(%__MODULE__{conn: conn, host: host, port: port}) do
    Agent.update(conn, fn old_socket ->
      safe_stop(old_socket)

      case open_socket(host, port) do
        {:ok, fresh_socket} -> fresh_socket
        {:error, _reason} -> old_socket
      end
    end)
  end

  defp safe_stop(socket) do
    if Process.alive?(socket) do
      GenServer.stop(socket, :normal, 1_000)
    end
  catch
    :exit, _ -> :ok
  end
end
