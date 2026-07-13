defmodule ControlLoop.ZeroMQClientTest do
  use ExUnit.Case, async: false

  alias ControlLoop.Test.FakeInferActionServer
  alias ControlLoop.ZeroMQClient

  @observation %{
    image: :binary.copy(<<0>>, 2 * 2 * 3),
    image_shape: {2, 2, 3},
    state: [0.1, 0.2, 0.3],
    instruction: "pick up the cube"
  }

  describe "wire format" do
    test "encode_request/1 matches the pinned schema (design.md component 01.3)" do
      encoded = ZeroMQClient.encode_request(@observation)
      {:ok, decoded} = Msgpax.unpack(encoded)

      assert %{
               "image" => image,
               "image_shape" => [2, 2, 3],
               "state" => [0.1, 0.2, 0.3],
               "instruction" => "pick up the cube"
             } = decoded

      assert image == @observation.image
    end

    test "decode_response/1 handles the ok shape" do
      raw =
        Msgpax.pack!(%{"ok" => %{"action_chunk" => [[1.0, 2.0], [3.0, 4.0]]}}, iodata: false)

      assert {:ok, [[1.0, 2.0], [3.0, 4.0]]} = ZeroMQClient.decode_response(raw)
    end

    test "decode_response/1 handles the error shape" do
      raw = Msgpax.pack!(%{"error" => %{"message" => "state too long"}}, iodata: false)
      assert {:error, {:server_error, "state too long"}} = ZeroMQClient.decode_response(raw)
    end
  end

  describe "infer_action/2 against a real (fake) REP peer" do
    setup do
      {:ok, server} = FakeInferActionServer.start_link()
      port = FakeInferActionServer.port(server)
      {:ok, client} = ZeroMQClient.connect(~c"localhost", port)
      %{server: server, client: client}
    end

    test "returns {:ok, action_chunk} on a canned ok response", %{client: client} do
      assert {:ok, [[1.0, 2.0], [3.0, 4.0]]} = ZeroMQClient.infer_action(client, @observation)
    end

    test "returns {:error, {:server_error, message}} on the error shape" do
      {:ok, server} =
        FakeInferActionServer.start_link(
          responder: fn _req -> %{"error" => %{"message" => "state too long"}} end
        )

      port = FakeInferActionServer.port(server)
      {:ok, client} = ZeroMQClient.connect(~c"localhost", port)

      assert {:error, {:server_error, "state too long"}} =
               ZeroMQClient.infer_action(client, @observation)
    end

    test "round trips a realistic multi-action chunk", %{server: server} do
      chunk = for i <- 1..50, do: for(j <- 1..32, do: i * 100 + j * 1.0)

      port = FakeInferActionServer.port(server)

      {:ok, server2} =
        FakeInferActionServer.start_link(
          responder: fn _req -> %{"ok" => %{"action_chunk" => chunk}} end
        )

      port2 = FakeInferActionServer.port(server2)
      {:ok, client} = ZeroMQClient.connect(~c"localhost", port2)

      assert {:ok, ^chunk} = ZeroMQClient.infer_action(client, @observation)
      # sanity: the original server/port from setup is untouched
      assert is_integer(port)
    end
  end

  describe "timeout handling" do
    test "returns {:error, :timeout} rather than blocking when the server never replies" do
      {:ok, server} = FakeInferActionServer.start_link(start_paused: true)
      port = FakeInferActionServer.port(server)

      {:ok, client} = ZeroMQClient.connect(~c"localhost", port, timeout_ms: 100)

      {elapsed_us, result} =
        :timer.tc(fn -> ZeroMQClient.infer_action(client, @observation) end)

      assert result == {:error, :timeout}
      # generous upper bound -- proves the call did not block indefinitely
      assert elapsed_us < 1_000_000
    end

    test "a later call still gets a real reply after a prior call timed out (recovers, doesn't stay wedged)" do
      {:ok, server} = FakeInferActionServer.start_link(start_paused: true)
      port = FakeInferActionServer.port(server)

      {:ok, client} = ZeroMQClient.connect(~c"localhost", port, timeout_ms: 100)
      assert {:error, :timeout} = ZeroMQClient.infer_action(client, @observation)

      :ok = FakeInferActionServer.resume(server)
      # give chumak's own peer-reconnect / our reopen a moment to settle
      Process.sleep(200)

      assert {:ok, [[1.0, 2.0], [3.0, 4.0]]} = ZeroMQClient.infer_action(client, @observation)
    end
  end

  describe "dropped connection" do
    test "a client connected before the server exists reconnects once the server starts" do
      port = Enum.random(20_000..29_999)
      {:ok, client} = ZeroMQClient.connect(~c"localhost", port, timeout_ms: 300)

      # server not up yet -- this call must fail cleanly, not hang or crash
      result = ZeroMQClient.infer_action(client, @observation)
      assert match?({:error, _reason}, result)

      {:ok, _server} = FakeInferActionServer.start_link(port: port)
      # chumak_peer retries the underlying TCP connect on a fixed 2000ms
      # timer (chumak.hrl's RECONNECT_TIMEOUT) -- give it enough headroom
      # to reconnect before asserting.
      Process.sleep(2_500)

      assert {:ok, [[1.0, 2.0], [3.0, 4.0]]} = ZeroMQClient.infer_action(client, @observation)
    end
  end
end
