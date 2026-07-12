# Elixir-native fine-tuning is the intended path; Python retires only on proven parity

<a id="adr-0005"></a>

Fine-tuning was originally designed Python-only (LeRobot's training recipe),
with only weights crossing to the Elixir-native inference adapter. We
reconsidered: keeping the whole system's runtime — inference, the control
loop, and now training — native to the BEAM has ecosystem-wide value for an
async, near-real-time robotics loop, and is worth pursuing even though
fine-tuning itself carries no hard latency constraint (unlike inference, a
training run's cost is minutes-to-hours, so a process hop would have been
tolerable). We considered running both trainers permanently side by side
(mirroring the inference adapters' shape exactly), but rejected that here:
unlike inference — where a numerical-parity prototype was a clean, achievable
bar — training quality has no equivalently clean numerical test; two correct
trainers can diverge in loss trajectory while converging to equally good
policies. The decision: Elixir-native fine-tuning (Nx/Axon) is the intended
target, cut over only when a task-performance-parity check (fine-tune both
trainers on identical episodes, compare the resulting policies' task success
rate on held-out evaluation episodes, not their loss curves) shows the
Elixir-native trainer is not meaningfully worse. Python fine-tuning is the
standing fallback — permanent, not throwaway — if that parity is never
reached; in that case, weights migrate from the Python trainer to the
Elixir-native inference adapter exactly as designed before this decision
([ADR-0004](0004-weights-only-cross-runtime-sharing.md#adr-0004)). Unlike the
inference adapters (ADR-0003, which keeps both permanently), this is a
one-way intended cutover with a named gate and a documented fallback — the
asymmetry is deliberate, not an oversight.
