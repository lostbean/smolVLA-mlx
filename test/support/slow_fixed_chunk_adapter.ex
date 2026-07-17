defmodule Demo.Test.SlowFixedChunkAdapter do
  @moduledoc """
  A deliberately SLOW variant of `Demo.Test.FixedChunkAdapter`, used only by
  the demo closed-loop test to prove the async cross-node `infer_action` call
  never blocks the tick loop (demo design 01.2 / criterion 4). It sleeps before
  delegating to the fixed-chunk stub, standing in for a heavy forward pass that
  takes real time.

  The delay lives in the adapter (the model seam), NOT in any demo code -- so
  the test proves ControlLoop's OWN async trigger keeps the tick loop draining
  while a call is in flight, with the slowness at the true external boundary.
  """

  @behaviour InferenceServer.Adapter

  @slow_ms 400

  @impl true
  def infer_action(model, observation) do
    Process.sleep(@slow_ms)
    Demo.Test.FixedChunkAdapter.infer_action(model, observation)
  end
end
