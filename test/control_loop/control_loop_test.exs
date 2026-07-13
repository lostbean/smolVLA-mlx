defmodule ControlLoop.ControlLoopTest do
  use ExUnit.Case, async: true

  alias ControlLoop.ActionQueue

  # A fake adapter matches ZeroMQClient.infer_action/2's own shape
  # (observation -> {:ok, action_chunk} | {:error, reason}) so ControlLoop
  # can be tested without any real network call -- dependency injection,
  # same pattern as the Python server chunk's injected model.
  defmodule FakeAdapter do
    def start_link(opts \\ []) do
      Agent.start_link(fn ->
        %{
          calls: 0,
          behavior: Keyword.get(opts, :behavior, fn -> {:ok, default_chunk()} end)
        }
      end)
    end

    def default_chunk, do: for(i <- 1..50, do: [i * 1.0])

    def infer_action(agent, _observation) do
      behavior =
        Agent.get_and_update(agent, fn state ->
          {state.behavior, %{state | calls: state.calls + 1}}
        end)

      # `behavior.()` runs outside the Agent's own process so a test
      # behavior that raises (simulating an adapter bug) crashes only the
      # calling Task, matching what a real ZeroMQClient.infer_action/2
      # bug would do -- it must not take the fake adapter's own bookkeeping
      # process down with it.
      behavior.()
    end

    def call_count(agent), do: Agent.get(agent, & &1.calls)

    def set_behavior(agent, fun) do
      Agent.update(agent, &%{&1 | behavior: fun})
    end
  end

  defp seed_queue(chunk_size \\ 50) do
    ActionQueue.new() |> ActionQueue.enqueue(for(i <- 1..chunk_size, do: [i * 1.0]))
  end

  describe "start_link/1" do
    test "accepts :zeromq_fallback and starts" do
      {:ok, adapter} = FakeAdapter.start_link()

      assert {:ok, pid} =
               ControlLoop.start_link(
                 adapter: :zeromq_fallback,
                 adapter_module: FakeAdapter,
                 adapter_client: adapter,
                 initial_queue: seed_queue(),
                 low_water_threshold: 25,
                 actuator_sink: fn _action -> :ok end
               )

      assert Process.alive?(pid)
    end

    test ":emily_native is accepted but not yet implemented" do
      assert {:error, {:not_yet_implemented, :emily_native}} =
               ControlLoop.start_link(adapter: :emily_native)
    end
  end

  describe "tick/1 -- queue_healthy state" do
    test "pops and sends one action per tick without calling infer_action" do
      {:ok, adapter} = FakeAdapter.start_link()
      test_pid = self()

      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(50),
          low_water_threshold: 10,
          actuator_sink: fn action -> send(test_pid, {:sent, action}) end
        )

      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [1.0]}
      assert FakeAdapter.call_count(adapter) == 0
    end
  end

  describe "tick/1 -- queue_low state" do
    test "pops, sends, and fires an async infer_action without blocking the tick" do
      {:ok, adapter} = FakeAdapter.start_link()
      test_pid = self()

      # threshold 45 with a 50-deep queue: after the first pop depth is 49,
      # still >= 45. Seed a smaller queue so the *pre-pop* depth is already
      # below threshold on this tick.
      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(10),
          low_water_threshold: 25,
          actuator_sink: fn action -> send(test_pid, {:sent, action}) end
        )

      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [1.0]}

      # the infer_action call must have been fired (async), not skipped
      Process.sleep(20)
      assert FakeAdapter.call_count(adapter) == 1
    end

    test "infer_action's result re-enters the queue via enqueue (aggregation)" do
      {:ok, adapter} = FakeAdapter.start_link()
      test_pid = self()

      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(5),
          low_water_threshold: 25,
          actuator_sink: fn action -> send(test_pid, {:sent, action}) end
        )

      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [1.0]}

      # wait for the async infer_action result to land back as an enqueue
      Process.sleep(50)
      assert ControlLoop.queue_depth(pid) == 4 + 50
    end

    test "does not fire a second infer_action while one is already in flight" do
      test_pid = self()

      {:ok, adapter} =
        FakeAdapter.start_link(
          behavior: fn ->
            send(test_pid, :infer_action_called)
            Process.sleep(200)
            {:ok, FakeAdapter.default_chunk()}
          end
        )

      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(10),
          low_water_threshold: 25,
          actuator_sink: fn _action -> :ok end
        )

      :ok = ControlLoop.tick(pid)
      assert_receive :infer_action_called

      # several more ticks fire while the first infer_action is still in
      # flight (200ms sleep) -- none of them should fire a second call.
      for _ <- 1..5, do: :ok = ControlLoop.tick(pid)
      refute_receive :infer_action_called, 150

      Process.sleep(150)
      assert FakeAdapter.call_count(adapter) == 1
    end
  end

  describe "a failed infer_action call" do
    test "leaves the queue draining on what it already has, without crashing or blocking ticks" do
      test_pid = self()

      {:ok, adapter} =
        FakeAdapter.start_link(behavior: fn -> {:error, :timeout} end)

      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(3),
          low_water_threshold: 25,
          actuator_sink: fn action -> send(test_pid, {:sent, action}) end
        )

      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [1.0]}
      Process.sleep(30)

      # the process is still alive and still able to drain the rest of the
      # queue that was already there.
      assert Process.alive?(pid)
      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [2.0]}
      assert Process.alive?(pid)
    end

    test "an adapter that raises does not wedge infer_action_in_flight forever" do
      test_pid = self()

      {:ok, adapter} = FakeAdapter.start_link(behavior: fn -> raise "boom" end)

      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(3),
          low_water_threshold: 25,
          actuator_sink: fn action -> send(test_pid, {:sent, action}) end
        )

      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [1.0]}
      Process.sleep(30)

      # a second infer_action must still be triggerable -- if the raise had
      # left infer_action_in_flight stuck at true, call_count would stay 1
      # forever.
      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [2.0]}
      Process.sleep(30)
      :ok = ControlLoop.tick(pid)
      Process.sleep(30)

      assert FakeAdapter.call_count(adapter) > 1
      assert Process.alive?(pid)
    end

    test "surfaces a degraded event when the queue empties before a result returns" do
      test_pid = self()

      {:ok, adapter} =
        FakeAdapter.start_link(
          behavior: fn ->
            Process.sleep(200)
            {:ok, FakeAdapter.default_chunk()}
          end
        )

      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(1),
          low_water_threshold: 25,
          actuator_sink: fn action -> send(test_pid, {:sent, action}) end,
          telemetry_sink: fn event -> send(test_pid, {:telemetry, event}) end
        )

      :ok = ControlLoop.tick(pid)
      assert_receive {:sent, [1.0]}

      # queue is now empty and no result has returned yet -- the next tick
      # must not crash and must surface the degraded condition rather than
      # silently doing nothing.
      :ok = ControlLoop.tick(pid)
      assert_receive {:telemetry, {:queue_exhausted, _depth}}
      assert Process.alive?(pid)
    end
  end

  describe "sustained run -- both foundation invariants hold" do
    test "the queue is never read past its safe depth, and no action is ever executed twice, across many ticks" do
      test_pid = self()
      chunk_size = 50
      low_water = 25
      chunk_counter = :counters.new(1, [])

      {:ok, adapter} =
        FakeAdapter.start_link(
          behavior: fn ->
            # simulate real network latency for a realistic interleaving
            Process.sleep(5)
            :counters.add(chunk_counter, 1, 1)
            chunk_id = :counters.get(chunk_counter, 1)
            # chunk_id makes every action from every distinct infer_action
            # call globally unique (no birthday-paradox collision risk, unlike
            # a random tag) -- an exact duplicate in sent_actions below can
            # only mean the *same* popped action was sent to the actuator
            # sink twice, which is exactly the invariant under test.
            {:ok, for(i <- 1..chunk_size, do: [chunk_id * 1.0, i * 1.0])}
          end
        )

      {:ok, pid} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: FakeAdapter,
          adapter_client: adapter,
          initial_queue: seed_queue(chunk_size),
          low_water_threshold: low_water,
          actuator_sink: fn action -> send(test_pid, {:sent, action}) end
        )

      total_ticks = 300

      sent_actions =
        for _ <- 1..total_ticks do
          :ok = ControlLoop.tick(pid)
          # let async infer_action results land between ticks, matching a
          # real tick-timer cadence and giving the queue a chance to refill
          Process.sleep(2)

          receive do
            {:sent, action} -> action
          after
            100 -> flunk("tick did not send an action -- queue was read past safe depth")
          end
        end

      # invariant: no action is ever executed (sent to the actuator sink)
      # twice. Actions are [chunk_id, i] pairs -- chunk_id is globally
      # unique per infer_action call (see chunk_counter above) and the
      # initial seed chunk uses i alone with no chunk_id collision possible
      # -- so exact list identity is a faithful, collision-free proxy for
      # "this exact action object was popped and sent more than once".
      assert length(sent_actions) == length(Enum.uniq(sent_actions))

      # invariant: the loop never crashed under sustained load, and never
      # silently stalled (every tick produced a send).
      assert Process.alive?(pid)
      assert length(sent_actions) == total_ticks
    end
  end
end
