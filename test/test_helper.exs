# Real-checkpoint tests (load the actual ~1.1GB lerobot/smolvla_base
# checkpoint and run real forward passes -- real wall-clock time) are
# excluded by default, mirroring model_runtime_server's own
# RUN_SMOLVLA_INTEGRATION_CHECK opt-in convention
# (model_runtime_server/tests/integration/test_server_real_checkpoint.py).
# Opt in with RUN_SMOLVLA_INTEGRATION_CHECK=1.
exclude =
  if System.get_env("RUN_SMOLVLA_INTEGRATION_CHECK") == "1" do
    []
  else
    [:real_checkpoint]
  end

ExUnit.start(exclude: exclude)
