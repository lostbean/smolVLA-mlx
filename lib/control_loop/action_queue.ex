defmodule ControlLoop.ActionQueue do
  @moduledoc """
  The ordered, currently-executing-plus-queued sequence of actions.

  Per `docs/design/control-loop/design.md` component 01.2: a plain,
  immutable data structure, not a process -- `ControlLoop` is the sole owner
  of every call site, per this context's "only ControlLoop reaches into the
  queue directly" invariant. `enqueue/2` always appends a newly-returned
  action chunk after whatever is still queued (aggregation, never
  replacement), matching SmolVLA's own reference queueing behavior.
  """

  defmodule EmptyQueueError do
    @moduledoc """
    Raised by `ControlLoop.ActionQueue.pop/1` when the queue is empty.

    Per component 01.2's "Fails" note: unreachable in practice under the
    low-water invariant (`ControlLoop` always tops up before the queue runs
    dry), but a caller bug that violates that invariant must never be masked
    by silently returning a no-op action.
    """
    defexception message: "ControlLoop.ActionQueue.pop/1 called on an empty queue"
  end

  @opaque t :: %__MODULE__{actions: :queue.queue()}
  defstruct actions: :queue.new()

  @doc "Returns a new, empty action queue."
  @spec new() :: t()
  def new, do: %__MODULE__{actions: :queue.new()}

  @doc """
  Appends `action_chunk` (a list of actions, e.g. SmolVLA's `(chunk_size,
  action_dim)` output) after whatever is still queued. Aggregation, never
  replacement.
  """
  @spec enqueue(t(), [term()]) :: t()
  def enqueue(%__MODULE__{actions: actions} = queue, action_chunk) when is_list(action_chunk) do
    %{queue | actions: :queue.join(actions, :queue.from_list(action_chunk))}
  end

  @doc """
  Pops the next action to execute, returning `{action, remaining_queue}`.

  Raises `ControlLoop.ActionQueue.EmptyQueueError` on an empty queue rather
  than returning a sentinel -- see the moduledoc and component 01.2's
  "Fails" note.
  """
  @spec pop(t()) :: {term(), t()}
  def pop(%__MODULE__{actions: actions} = queue) do
    case :queue.out(actions) do
      {{:value, action}, remaining} -> {action, %{queue | actions: remaining}}
      {:empty, _} -> raise EmptyQueueError
    end
  end

  @doc "The number of not-yet-executed actions currently queued."
  @spec depth(t()) :: non_neg_integer()
  def depth(%__MODULE__{actions: actions}), do: :queue.len(actions)
end
