defmodule SimEnvAdapterTest do
  @moduledoc """
  Fast-suite tests for `SimEnvAdapter` (demo design component 01.1).

  These assert behavior through the adapter's PUBLIC interface only
  (`observe/1`, `actuate/2`, and the two ControlLoop closures) -- never the
  adapter's private transport state. The Python `sim_server` (real MuJoCo) is
  a TRUE external, so the wire logic is exercised against a real ZeroMQ REP
  peer -- `ControlLoop.Test.FakeSimServer` -- that returns canned MessagePack
  replies, mirroring how `ControlLoop.ZeroMQClientTest` tests `ZeroMQClient`
  against `FakeInferActionServer`. The real-MuJoCo path is a separate, gated
  test (see `sim_env_adapter_real_sim_test.exs`).
  """
  use ExUnit.Case, async: false

  alias ControlLoop.Test.FakeSimServer

  @instruction "pick up the cube and place it on the target"

  # A full 32-dim SmolVLA action chunk action; the adapter must slice the
  # leading 6 before it reaches the sim's 6-DoF step.
  @action_32 for i <- 1..32, do: i * 1.0

  defp start_adapter(server, opts \\ []) do
    port = FakeSimServer.port(server)

    {:ok, adapter} =
      SimEnvAdapter.start_link(
        Keyword.merge(
          [address: "tcp://localhost:#{port}", instruction: @instruction],
          opts
        )
      )

    adapter
  end

  describe "start_link/1" do
    test "connects and resets the env at start" do
      {:ok, server} = FakeSimServer.start_link()
      _adapter = start_adapter(server)

      # The very first request the adapter issues must be a reset.
      assert [%{"op" => "reset"} | _] = FakeSimServer.requests(server)
    end

    test "fails loud when the sim server is unreachable at start" do
      # No server on this port -- start must not silently succeed. init/1
      # returns {:stop, reason}, so start_link/1 returns {:error, reason}; the
      # linked stop also signals the caller, so trap exits to observe the
      # explicit error return cleanly.
      Process.flag(:trap_exit, true)
      port = Enum.random(30_000..39_999)

      result =
        SimEnvAdapter.start_link(
          address: "tcp://localhost:#{port}",
          instruction: @instruction,
          # small timeout so the failing initial reset returns fast
          timeout_ms: 300
        )

      assert {:error, {:sim_server_unreachable, _reason}} = result
    end
  end

  describe "observe/1" do
    test "returns a well-formed observation with the fixed instruction and 6-DoF state" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)

      obs = SimEnvAdapter.observe(adapter)

      assert %{
               image: image,
               image_shape: {h, w, c},
               state: state,
               instruction: @instruction
             } = obs

      assert is_binary(image)
      assert {h, w, c} == {4, 4, 3}
      assert byte_size(image) == h * w * c
      assert is_list(state)
      assert length(state) == 6
      assert Enum.all?(state, &is_float/1)
    end
  end

  describe "actuate/2 advances the sim (the arm moves)" do
    test "the next observe reflects the movement" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)

      before = SimEnvAdapter.observe(adapter)
      assert :ok = SimEnvAdapter.actuate(adapter, @action_32)
      after_move = SimEnvAdapter.observe(adapter)

      # The observation genuinely changed -- the sim advanced.
      refute after_move.state == before.state
      refute after_move.image == before.image
    end

    test "issues a step whose action is the FIRST 6 elements of the 32-dim action" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)

      :ok = SimEnvAdapter.actuate(adapter, @action_32)

      step = Enum.find(FakeSimServer.requests(server), &match?(%{"op" => "step"}, &1))
      assert %{"op" => "step", "action" => action} = step
      assert action == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "a malformed (too-short) action fails loud, never padded or swallowed" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)

      # Fewer than 6 elements is malformed per the take-first-6 contract.
      assert_raise ArgumentError, fn ->
        SimEnvAdapter.actuate(adapter, [1.0, 2.0, 3.0])
      end
    end
  end

  describe "the two ControlLoop closures" do
    test "observation_source/1 is a zero-arity function returning an observation" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)

      source = SimEnvAdapter.observation_source(adapter)
      assert is_function(source, 0)

      obs = source.()
      assert %{image: _, image_shape: {_, _, _}, state: _, instruction: @instruction} = obs
    end

    test "actuator_sink/1 is a (action -> :ok) function that advances the sim" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)

      sink = SimEnvAdapter.actuator_sink(adapter)
      assert is_function(sink, 1)

      before = SimEnvAdapter.observe(adapter)
      assert :ok = sink.(@action_32)
      after_move = SimEnvAdapter.observe(adapter)

      refute after_move.state == before.state
    end

    test "the closures drive ControlLoop unmodified beyond wiring the two options" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)
      test_pid = self()

      # Prove the exact seam ControlLoop expects: a zero-arity source and a
      # (action -> _) sink, wired via ControlLoop's own options with no
      # demo-specific change to ControlLoop.
      {:ok, loop} =
        ControlLoop.start_link(
          adapter: :zeromq_fallback,
          adapter_module: __MODULE__.StubAdapter,
          adapter_client: :ignored,
          initial_queue:
            ControlLoop.ActionQueue.new() |> ControlLoop.ActionQueue.enqueue([@action_32]),
          observation_source: SimEnvAdapter.observation_source(adapter),
          actuator_sink: fn action ->
            SimEnvAdapter.actuator_sink(adapter).(action)
            send(test_pid, {:actuated, action})
          end
        )

      before = SimEnvAdapter.observe(adapter)
      :ok = ControlLoop.tick(loop)
      # ControlLoop popped the action and drove it through the sim sink.
      assert_receive {:actuated, @action_32}
      after_move = SimEnvAdapter.observe(adapter)
      refute after_move.state == before.state
    end
  end

  describe "demoable standalone: a canned sequence drives the sim, arm motion is observable" do
    test "interleaved actuate/observe with no ControlLoop present" do
      {:ok, server} = FakeSimServer.start_link()
      adapter = start_adapter(server)

      # A canned sequence of actions; record the state after each.
      states =
        for k <- 1..5 do
          action = for i <- 1..32, do: (k * 100 + i) * 1.0
          :ok = SimEnvAdapter.actuate(adapter, action)
          SimEnvAdapter.observe(adapter).state
        end

      # The arm's state genuinely evolves over the sequence -- each step moved it.
      assert length(Enum.uniq(states)) == 5
    end
  end

  describe "fail loud on a sim-server error reply" do
    # A responder that lets the initial reset succeed (so start_link works)
    # but errors on every other op -- isolating the observe/actuate error path.
    defp reset_ok_then_error(message) do
      fn
        %{"op" => "reset"}, _tick ->
          %{
            "ok" => %{
              "image" => %Msgpax.Bin{data: :binary.copy(<<0>>, 48)},
              "image_shape" => [4, 4, 3],
              "state" => List.duplicate(0.0, 6)
            }
          }

        _other, _tick ->
          %{"error" => %{"message" => message}}
      end
    end

    test "a step error surfaces loud via actuate/2 and observe/1 is never left with a fabricated frame" do
      # observe/1 returns the observation coupled to the last real reset/step
      # (design invariant: never a frame divorced from the arm's real state).
      # A failing step surfaces loud through actuate/2, and observe/1 still
      # returns the last GOOD observation -- it never fabricates a blank frame.
      {:ok, server} = FakeSimServer.start_link(responder: reset_ok_then_error("env raised"))
      adapter = start_adapter(server)

      good = SimEnvAdapter.observe(adapter)

      assert_raise RuntimeError, ~r/env raised/, fn ->
        SimEnvAdapter.actuate(adapter, @action_32)
      end

      assert SimEnvAdapter.observe(adapter) == good
    end

    test "actuate/2 surfaces an env error rather than silently dropping the action" do
      {:ok, server} = FakeSimServer.start_link(responder: reset_ok_then_error("bad action shape"))
      adapter = start_adapter(server)

      assert_raise RuntimeError, ~r/bad action shape/, fn ->
        SimEnvAdapter.actuate(adapter, @action_32)
      end
    end
  end

  # A no-op adapter so ControlLoop's infer_action trigger path doesn't reach a
  # real model in the closures test -- this test is about the obs/actuator
  # seams, not inference.
  defmodule StubAdapter do
    def infer_action(_client, _observation), do: {:ok, []}
  end
end
