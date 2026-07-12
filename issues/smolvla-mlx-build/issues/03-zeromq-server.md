Status: ready-for-agent
Category: enhancement

## Parent

[control-loop design](../../../docs/design/control-loop/design.md), component 01.3 (`ZeroMQ client`, server side). Pending build entry, [ADR-0002](../../../docs/adr/0002-elixir-owns-the-control-loop.md#adr-0002) and [ADR-0003](../../../docs/adr/0003-emily-native-primary-zeromq-fallback.md#adr-0003).

## What to build

Wrap the Python `infer_action` call (from the previous slice) in a ZeroMQ
server: it accepts one `{#term-observation}` per request over the network
(LAN or same-host — the design never assumes colocated deployment) and
returns one `{#term-action-chunk}` per response.

This is the server side of the permanent Python fallback adapter for the
`{#term-infer-action-port}` — designed to run as a standing service on this
Mac, reachable by a Raspberry Pi or other Elixir cluster node elsewhere on
the network. No new inference logic here; this slice is purely the
request/response wrapper around the previous slice's `infer_action`.

## Acceptance criteria

- [ ] A running server process accepts one `infer_action`-shaped request and
      returns one action-chunk-shaped response over ZeroMQ.
- [ ] The server is reachable from a separate process on the same machine
      (proving the network path works before a real second machine is
      available for testing).
- [ ] A malformed request (wrong shape, wrong dimensionality) is rejected
      with an explicit error response, never silently coerced or dropped.
- [ ] The server survives and continues serving subsequent requests after a
      client disconnects mid-request.

## Out of scope

The Elixir-side client (next slice); the emily-native adapter; fine-tuning;
authentication or encryption on the ZeroMQ channel (not a stated invariant
in the current design — flag as a follow-up if the deployment context needs
it).

## Blocked by

02-infer-action-python
