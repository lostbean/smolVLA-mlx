defmodule ControlLoop.Test.FakeSimServer do
  @moduledoc """
  A tiny, real `chumak`-based REP server standing in for the Python
  `sim_server` process in fast tests, per this repo's seam-testing doctrine
  (a real lightweight stand-in for a "remote-but-owned service", not a mock
  -- mirrors `ControlLoop.Test.FakeInferActionServer`).

  It speaks the sim server's wire contract (see
  `sim_server/server.py` / demo design 01.1): MessagePack with STRING keys,
  answering `reset` / `step` / `render` with `{"ok" => payload}` or
  `{"error" => %{"message" => msg}}`. By default it acts like a real gym env:
  it holds a monotonically advancing "tick" so that each `reset`/`step`
  yields a DIFFERENT frame and state (a `step` advances; a `render` does
  not), letting a test prove the arm "moves" through the adapter. The
  requests it received are recorded so a test can assert what actions were
  forwarded on the wire.

  A custom `responder` (a `(decoded_request, tick -> encodable_reply)`
  function) overrides the default env behavior for error-path tests.

  `:chumak.recv/1` blocks the calling process until a request arrives, so the
  accept/reply loop runs on its own linked process (same rationale as
  `FakeInferActionServer`).
  """

  use GenServer

  @image_shape [4, 4, 3]

  def start_link(opts \\ []) do
    responder = Keyword.get(opts, :responder, nil)
    port = Keyword.get(opts, :port, random_test_port())
    start_paused = Keyword.get(opts, :start_paused, false)
    GenServer.start_link(__MODULE__, {responder, port, start_paused})
  end

  defp random_test_port, do: Enum.random(20_000..29_999)

  @doc "The port this fake server actually bound to."
  def port(pid), do: GenServer.call(pid, :port)

  @doc "The list of decoded requests this server has received, oldest first."
  def requests(pid), do: GenServer.call(pid, :requests)

  @impl true
  def init({responder, port, start_paused}) do
    {:ok, socket} = :chumak.socket(:rep)
    {:ok, _bind_pid} = :chumak.bind(socket, :tcp, ~c"localhost", port)

    owner = self()
    loop_pid = spawn_link(fn -> loop(socket, responder, start_paused, 0, owner) end)

    {:ok, %{socket: socket, loop_pid: loop_pid, port: port, requests: []}}
  end

  @impl true
  def handle_call(:port, _from, state), do: {:reply, state.port, state}
  def handle_call(:requests, _from, state), do: {:reply, Enum.reverse(state.requests), state}

  @impl true
  def handle_info({:request, request}, state) do
    {:noreply, %{state | requests: [request | state.requests]}}
  end

  # Runs on its own process: blocks in :chumak.recv/1, replies via the
  # default gym-like env behavior (a stateful tick) unless a custom
  # responder is supplied.
  defp loop(socket, responder, paused, tick, owner) do
    if paused do
      Process.sleep(20)
      loop(socket, responder, paused, tick, owner)
    else
      case :chumak.recv(socket) do
        {:ok, raw_request} ->
          {:ok, request} = Msgpax.unpack(raw_request)
          send(owner, {:request, request})
          {reply, next_tick} = respond(request, responder, tick)
          :ok = :chumak.send(socket, Msgpax.pack!(reply, iodata: false))
          loop(socket, responder, paused, next_tick, owner)

        {:error, _reason} ->
          loop(socket, responder, paused, tick, owner)
      end
    end
  end

  # Custom responder: full control over the reply; it does not advance the
  # tick (error-path tests don't need a moving env).
  defp respond(request, responder, tick) when is_function(responder, 2) do
    {responder.(request, tick), tick}
  end

  # Default: behave like a real gym env over the wire.
  defp respond(%{"op" => "reset"}, _responder, _tick) do
    {%{"ok" => observation_payload(0)}, 0}
  end

  defp respond(%{"op" => "step", "action" => action}, _responder, tick)
       when is_list(action) do
    next = tick + 1
    {%{"ok" => observation_payload(next)}, next}
  end

  defp respond(%{"op" => "render"}, _responder, tick) do
    # render does NOT advance the sim; image only, no state.
    {%{"ok" => %{"image" => frame_bytes(tick), "image_shape" => @image_shape}}, tick}
  end

  defp respond(other, _responder, tick) do
    {%{"error" => %{"message" => "unknown op: #{inspect(other)}"}}, tick}
  end

  # A frame + 6-DoF state whose content depends on the tick, so consecutive
  # steps differ -- lets a test prove the observation changes as the arm moves.
  defp observation_payload(tick) do
    %{
      "image" => frame_bytes(tick),
      "image_shape" => @image_shape,
      "state" => List.duplicate(tick * 1.0, 6)
    }
  end

  defp frame_bytes(tick) do
    [h, w, c] = @image_shape
    %Msgpax.Bin{data: :binary.copy(<<rem(tick, 256)>>, h * w * c)}
  end
end
