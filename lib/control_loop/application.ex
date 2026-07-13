defmodule ControlLoop.Application do
  @moduledoc """
  OTP application entry point for the control-loop side of the system (see
  `docs/design/control-loop/design.md`).

  This chunk does not stand up a fixed supervision tree of its own --
  `ControlLoop` GenServers are started per-caller via `ControlLoop.start_link/1`
  (e.g. by a future bb bot supervisor, or directly in tests) rather than
  under a hardcoded child spec here. This module exists so `:control_loop`
  is a well-formed OTP application whose runtime dependencies (`:chumak`)
  are started automatically by Mix's application boot order.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = []
    Supervisor.start_link(children, strategy: :one_for_one, name: ControlLoop.Supervisor)
  end
end
