defmodule ControlLoop.ZeroMQClient do
  @moduledoc """
  The ZeroMQ fallback adapter's client half: sends one `infer_action`
  request to the Python `model-runtime` process over a `chumak` REQ socket
  and returns its decoded response.

  Per `docs/design/control-loop/design.md` component 01.3 and
  `docs/adr/0007-msgpack-wire-format-for-zeromq-fallback.md` /
  `docs/adr/0008-chumak-pure-erlang-zeromq-client.md`:

    * the wire format is MessagePack over a ZeroMQ REQ/REP socket -- see
      `encode_request/1` and `decode_response/1` for the exact shape;
    * `msgpax` decodes map keys as strings, never atoms, so responses are
      pattern-matched on `"ok"` / `"error"`, not atoms (ADR-0008);
    * a dropped connection reconnects rather than crashing the caller --
      `chumak`'s own peer process already retries the underlying TCP
      connection on a timer once `connect/4` has been called once (see
      `chumak_peer`'s `tcp_closed` handling), so this module does not need
      to redial on every call. What it *does* need to guard is the REQ
      socket's own request/reply state machine getting stuck
      (`{:error, :efsm}` from `chumak_req`) after an abandoned call -- e.g.
      one this client gave up on via a timeout, which leaves the old socket
      parked mid-cycle waiting for a reply nothing will ever collect. Since
      `infer_action/2`'s public shape (`client, observation ->
      {:ok, chunk} | {:error, reason}`) never hands back an updated client,
      the mutable "which chumak socket pid is live right now" cell has to
      live *inside* `t()` without changing `t()`'s own identity -- this
      module keeps that cell in a small owned `Agent` (`t().conn`) so a
      wedged socket can be closed and replaced transparently between calls;
    * every call carries a timeout (`chumak:recv/1` has no built-in one --
      internally a `gen_server:call(..., :infinity)` -- so a hung Python
      process would otherwise block the calling process forever). This
      module wraps the send+recv round trip in a supervised `Task` and
      converts a `Task.yield` timeout into `{:error, :timeout}`;
    * fails loud and local: a timeout or connection failure always returns
      `{:error, reason}` -- never an indefinite retry, never a swallowed
      error. Deciding what a failed call means for a tick is `ControlLoop`'s
      job, not this module's.
  """

  require Logger

  defstruct [:conn, :host, :port, :timeout_ms]

  @type observation :: %{
          required(:image) => binary(),
          required(:image_shape) => {non_neg_integer(), non_neg_integer(), non_neg_integer()},
          required(:state) => [float()],
          required(:instruction) => String.t()
        }

  @type action_chunk :: [[float()]]

  @type t :: %__MODULE__{
          conn: pid(),
          host: charlist() | String.t(),
          port: non_neg_integer(),
          timeout_ms: pos_integer()
        }

  @default_timeout_ms 5_000

  @doc """
  Opens a REQ socket and connects it to the Python `infer_action` server at
  `host:port`. `timeout_ms` bounds every subsequent `infer_action/2` call
  (default #{@default_timeout_ms}ms).
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
  Sends one `infer_action` request for `observation` and returns
  `{:ok, action_chunk}` or `{:error, reason}`.

  `reason` is one of:
    * `:timeout` -- the round trip did not complete within the client's
      configured timeout;
    * `{:server_error, message}` -- the Python server itself rejected the
      request or failed inference (wire-format `error` response);
    * any other term surfaced by `chumak` (e.g. `:no_connected_peers` while
      a dropped connection is being re-established in the background,
      `:closed`, `:efsm`).

  Never raises and never blocks past the client's configured timeout.
  """
  @spec infer_action(t(), observation()) :: {:ok, action_chunk()} | {:error, term()}
  def infer_action(%__MODULE__{} = client, observation) do
    request = encode_request(observation)
    socket = Agent.get(client.conn, & &1)

    task =
      Task.async(fn ->
        with :ok <- :chumak.send(socket, request),
             {:ok, raw_response} <- :chumak.recv(socket) do
          decode_response(raw_response)
        end
      end)

    case Task.yield(task, client.timeout_ms) || Task.shutdown(task, :brutal_kill) do
      {:ok, {:error, :efsm}} ->
        Logger.warning("ControlLoop.ZeroMQClient: REQ socket wedged (:efsm), reopening")
        reopen(client)
        {:error, :efsm}

      {:ok, result} ->
        result

      nil ->
        Logger.warning(
          "ControlLoop.ZeroMQClient: infer_action timed out after #{client.timeout_ms}ms"
        )

        reopen(client)
        {:error, :timeout}

      {:exit, reason} ->
        {:error, {:task_exit, reason}}
    end
  end

  defp open_socket(host, port) do
    with {:ok, socket} <- :chumak.socket(:req),
         {:ok, _peer} <- :chumak.connect(socket, :tcp, to_charlist(host), port) do
      {:ok, socket}
    end
  end

  # A timed-out or `:efsm`-wedged call leaves the REQ socket's own
  # request/reply state machine stuck mid-cycle (chumak's `chumak_req`
  # refuses a new `send/2` until the prior request's `recv/1` completes).
  # Closing that socket and opening a fresh one resets the state machine;
  # the underlying TCP connection is re-established by chumak's own
  # peer-level retry as soon as `connect/4` runs again, so this recovery
  # step is cheap and does not need its own backoff loop -- `ControlLoop`
  # simply sees this call's result as `{:error, reason}` and tries again on
  # its own next low-water trigger.
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

  @doc false
  @spec encode_request(observation()) :: binary()
  def encode_request(%{
        image: image,
        image_shape: {height, width, channels},
        state: state,
        instruction: instruction
      }) do
    Msgpax.pack!(
      %{
        "image" => %Msgpax.Bin{data: image},
        "image_shape" => [height, width, channels],
        "state" => Enum.map(state, &(&1 * 1.0)),
        "instruction" => instruction
      },
      iodata: false
    )
  end

  @doc false
  @spec decode_response(binary()) :: {:ok, action_chunk()} | {:error, term()}
  def decode_response(raw_response) do
    case Msgpax.unpack(raw_response) do
      {:ok, %{"ok" => %{"action_chunk" => action_chunk}}} ->
        {:ok, action_chunk}

      {:ok, %{"error" => %{"message" => message}}} ->
        {:error, {:server_error, message}}

      {:ok, other} ->
        {:error, {:malformed_response, other}}

      {:error, reason} ->
        {:error, {:malformed_response, reason}}
    end
  end
end
