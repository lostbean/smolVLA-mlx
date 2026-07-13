defmodule ControlLoop.Test.FakeInferActionServer do
  @moduledoc """
  A tiny, real `chumak`-based REP server standing in for the Python
  `model-runtime` process in fast tests, per this repo's seam-testing
  doctrine (a real lightweight stand-in for a "remote-but-owned service",
  not a mock).

  Started per-test on a random high port; `responder` decides what to reply
  to each decoded MessagePack request (a raw map with string keys, matching
  what the real Python server would send/receive on the wire).

  `:chumak.recv/1` blocks the calling process until a request arrives, so
  the actual accept/reply loop runs on its own linked process -- keeping it
  inline in this module's own GenServer would wedge `handle_call/3` (e.g.
  `port/1`, `pause/1`) behind whatever `recv/1` is doing.
  """

  use GenServer

  @doc """
  Starts the fake server on `port` (a random high port picked by the caller
  by default -- chumak's `bind/4` does not expose the OS-resolved port when
  given `0`, unlike a raw `gen_tcp` listen socket, so this helper avoids
  ephemeral-port resolution entirely) running `responder`, a
  `(decoded_request -> encodable_reply)` function.

  `start_paused: true` starts the server without ever calling `:chumak.recv/1`
  in the first place -- use this (rather than racing a `pause/1` call
  against the loop process's first blocking `recv`) when a test needs the
  server to be silent from the very first request.
  """
  def start_link(opts \\ []) do
    responder = Keyword.get(opts, :responder, &default_responder/1)
    port = Keyword.get(opts, :port, random_test_port())
    start_paused = Keyword.get(opts, :start_paused, false)
    GenServer.start_link(__MODULE__, {responder, port, start_paused})
  end

  defp random_test_port, do: Enum.random(20_000..29_999)

  @doc "The port this fake server actually bound to."
  def port(pid), do: GenServer.call(pid, :port)

  @doc """
  Stops replying to requests once any in-flight `recv/1` resolves. Blocks
  until the loop process has acknowledged the pause, so a caller never races
  a subsequent client call against the loop's next iteration -- prefer
  `start_link(start_paused: true)` when the server must be silent from its
  very first request, since a request that arrives before this call's ack
  will still be answered.
  """
  def pause(pid) do
    loop_pid = GenServer.call(pid, :loop_pid)
    send(loop_pid, {:pause, self()})

    receive do
      :paused -> :ok
    after
      1_000 -> {:error, :ack_timeout}
    end
  end

  @doc "Resumes replying after `pause/1`."
  def resume(pid) do
    loop_pid = GenServer.call(pid, :loop_pid)
    send(loop_pid, :resume)
    :ok
  end

  defp default_responder(_request) do
    %{"ok" => %{"action_chunk" => [[1.0, 2.0], [3.0, 4.0]]}}
  end

  @impl true
  def init({responder, port, start_paused}) do
    {:ok, socket} = :chumak.socket(:rep)
    {:ok, _bind_pid} = :chumak.bind(socket, :tcp, ~c"localhost", port)

    loop_pid = spawn_link(fn -> loop(socket, responder, start_paused) end)

    {:ok, %{socket: socket, loop_pid: loop_pid, port: port}}
  end

  @impl true
  def handle_call(:port, _from, state), do: {:reply, state.port, state}
  def handle_call(:loop_pid, _from, state), do: {:reply, state.loop_pid, state}

  # Runs on its own process: blocks in :chumak.recv/1 (unavoidable -- chumak
  # exposes no non-blocking/poll variant), replies via `responder` unless
  # paused. While paused the loop never calls recv again, so the client's
  # own `recv/1` genuinely hangs -- matching a Python process wedged
  # mid-inference -- rather than resolving with an error reply (which would
  # complete the client's REQ/REP cycle immediately and defeat the point of
  # a timeout test).
  defp loop(socket, responder, paused) do
    receive do
      {:pause, ack_to} ->
        send(ack_to, :paused)
        loop(socket, responder, true)

      :resume ->
        loop(socket, responder, false)
    after
      0 ->
        if paused do
          Process.sleep(20)
          loop(socket, responder, paused)
        else
          case :chumak.recv(socket) do
            {:ok, raw_request} ->
              {:ok, request} = Msgpax.unpack(raw_request)
              reply = responder.(request)
              :ok = :chumak.send(socket, Msgpax.pack!(reply, iodata: false))

            {:error, _reason} ->
              :ok
          end

          loop(socket, responder, paused)
        end
    end
  end
end
