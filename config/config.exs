import Config

# The emily-native infer_action adapter (ADR-0003, model-runtime design
# component 01.2) is the primary Nx backend/compiler for this project.
# `:raise` on an op that falls back to Nx.BinaryBackend rather than a
# native MLX kernel, so a silent slow-path never masks itself during
# development -- see docs/design/model-runtime/design.md component 01.2's
# de-risking notes.
config :emily, fallback: :raise

import_config "#{config_env()}.exs"
