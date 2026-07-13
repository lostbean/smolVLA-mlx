defmodule ControlLoop.ActionQueueTest do
  use ExUnit.Case, async: true

  alias ControlLoop.ActionQueue

  describe "new/0" do
    test "starts empty" do
      assert ActionQueue.new() |> ActionQueue.depth() == 0
    end
  end

  describe "enqueue/2 and pop/1" do
    test "pop returns actions in FIFO order" do
      queue =
        ActionQueue.new()
        |> ActionQueue.enqueue([[1.0, 2.0], [3.0, 4.0]])

      {action, queue} = ActionQueue.pop(queue)
      assert action == [1.0, 2.0]

      {action, queue} = ActionQueue.pop(queue)
      assert action == [3.0, 4.0]

      assert ActionQueue.depth(queue) == 0
    end

    test "enqueue appends a new chunk after whatever is still queued (aggregation, not replacement)" do
      queue =
        ActionQueue.new()
        |> ActionQueue.enqueue([[1.0], [2.0], [3.0]])

      {_action, queue} = ActionQueue.pop(queue)
      assert ActionQueue.depth(queue) == 2

      # a second chunk arrives while [2.0, 3.0] is still queued -- it must be
      # appended after them, never replace them.
      queue = ActionQueue.enqueue(queue, [[10.0], [11.0]])
      assert ActionQueue.depth(queue) == 4

      {a1, queue} = ActionQueue.pop(queue)
      {a2, queue} = ActionQueue.pop(queue)
      {a3, queue} = ActionQueue.pop(queue)
      {a4, queue} = ActionQueue.pop(queue)

      assert [a1, a2, a3, a4] == [[2.0], [3.0], [10.0], [11.0]]
      assert ActionQueue.depth(queue) == 0
    end

    test "depth reflects the number of not-yet-executed actions" do
      queue =
        ActionQueue.new()
        |> ActionQueue.enqueue([[1.0], [2.0], [3.0], [4.0], [5.0]])

      assert ActionQueue.depth(queue) == 5
    end
  end

  describe "pop/1 on an empty queue" do
    test "raises rather than silently returning a no-op action" do
      assert_raise ControlLoop.ActionQueue.EmptyQueueError, fn ->
        ActionQueue.new() |> ActionQueue.pop()
      end
    end

    test "raises again after being drained to empty" do
      queue =
        ActionQueue.new()
        |> ActionQueue.enqueue([[1.0]])

      {_action, queue} = ActionQueue.pop(queue)
      assert ActionQueue.depth(queue) == 0

      assert_raise ControlLoop.ActionQueue.EmptyQueueError, fn ->
        ActionQueue.pop(queue)
      end
    end
  end
end
