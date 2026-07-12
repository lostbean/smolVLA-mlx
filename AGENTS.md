# AGENTS.md

<!-- agent-skills:begin -->

(machine-owned; do not edit inside this fence — re-run setup to refresh)

## Agent skills

**Design layer** — `CONTEXT-MAP.md` indexes the design documents
(`design.md`, rendered `design.html`); terms are defined in each context's
`CONTEXT.md`; decisions are recorded in `docs/adr/`.

**Tracker** — local markdown under `issues/`: each feature lives at
`issues/<feature-slug>/`, with `PRD.md` and `issues/<NN>-<slug>.md` files
numbered from 01. Labels are the `Status:`/`Category:` header lines on each
issue file, using the canonical role names directly as their values:
`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix` (status); `bug`, `enhancement` (category).

**AI disclaimer** — every AI-authored tracker comment starts with:
`[AI-authored]`.

**Design gate** — `scripts/design-render --check <each design.md>` and
`scripts/layer-integrity .` check the design layer (exit 0 clean, 1
violation, 2 error).

**Staleness** — if the system has moved many commits since the design
documents last changed, reconcile design and code before relying on the
layer.

<!-- agent-skills:end -->

## Conventions

Clean commit messages — no trailers, no attribution, no Co-Authored-By, no
"Generated with" footers.

## Tooling

This repo uses a nix-native dev environment.

- `nix develop` (or `direnv allow`, since `.envrc` runs `use flake`) — opens a
  shell with Python (`python3`, `uv`) and Elixir (`elixir`), plus `lefthook`.
- `nix fmt` — formats the whole repo (`ruff-format` for Python, `mix-format`
  for Elixir, `nixfmt` for Nix) via treefmt.
- `nix flake check` — fails if the tree is not formatted.
- Pre-commit (`lefthook install` from inside the shell) formats staged files
  and re-stages them, then runs the design-layer gate:
  `scripts/design-render --check` on every `design.md`, then
  `scripts/layer-integrity .`.

## Status

The Elixir-native inference path (`emily`/`Nx.Defn`, see
[model-runtime](docs/design/model-runtime/design.md) component 01.2) is
prototype-verified: the core mechanism (backbone self-attention, the
flow-matching action expert's cross-attention into a frozen intermediate
layer, multi-step Euler integration) matches a NumPy reference to float32
precision and runs at ~5ms p50 — well under the 100ms budget. The full-scale
port against SmolVLA's real weights is still unbuilt (see the pending ledger
in [docs/design/design.md](docs/design/design.md)). `emily` installs from Hex
with a precompiled NIF — no MLX/C++ build step needed on this Mac.
