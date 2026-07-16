# BEAM distribution is a topology axis orthogonal to the infer_action port, not a third adapter

<a id="adr-0010"></a>

Supersedes the "removes a process/network hop" framing in
[ADR-0003](0003-emily-native-primary-zeromq-fallback.md#adr-0003) where it
implied same-host operation. The NERVES demo runs the emily-native adapter
(model-runtime component 01.2) on the Mac and calls it from a Raspberry Pi
node across a BEAM cluster — a network hop that ADR-0003's "no hop" argument
did not anticipate. We considered making the distributed call a third adapter
behind the `infer_action` port, alongside in-process emily and ZeroMQ-Python,
but rejected it: it would multiply adapters against the "one contract, not two
features" invariant and braid transport into the port. Instead, **where a node
running an adapter sits is a separate deployment-topology axis, orthogonal to
which adapter answers the port**. The emily-native adapter stays exactly one
adapter; a Mac-side long-lived process holds the loaded model and answers
`infer_action` calls, reachable in-process or across the cluster by native
BEAM distribution (`GenServer.call` to a named process on a remote node) with
no code or serialization-format change at the call site. ADR-0003's
load-bearing claim is therefore re-scoped: emily-native's advantage over the
ZeroMQ-Python fallback is **no foreign serialization hop** — native BEAM term
passing rather than MessagePack-over-ZeroMQ
([ADR-0007](0007-msgpack-wire-format-for-zeromq-fallback.md#adr-0007)), and no
separate OS process — *not* "same physical host," which was never the real
property. Both adapters may now cross a network; the emily-native one does so
without leaving the BEAM or encoding to a foreign wire format.
