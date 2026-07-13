# The Elixir ZeroMQ client is chumak, a pure-Erlang ZMTP implementation, not a libzmq NIF binding

<a id="adr-0008"></a>

The `ZeroMQClient` (control-loop component 01.3) needed a real Hex library.
We considered `erlzmq_dnif`, an actively maintained NIF binding to the real
libzmq C library using dirty NIFs, since it is the more conventional choice
and gives access to libzmq's full feature set. We chose `chumak` instead: a
pure-Erlang reimplementation of the ZMTP 3.1 wire protocol (interoperable
with libzmq's own ZMTP 3.x, including Python's `pyzmq`), maintained under the
official `zeromq` GitHub organization, with explicit REQ/REP support and its
own acceptance tests. Choosing chumak means this client needs no NIF and no
libzmq system library at all — the same property that makes the emily-native
inference adapter attractive over a Python subprocess
([ADR-0003](0003-emily-native-primary-zeromq-fallback.md#adr-0003)) applies
here too: the Raspberry Pi that will eventually run this code gets one fewer
native-build dependency to cross-compile or install. chumak's REQ socket has
no built-in retry or timeout — same as libzmq's own REQ, so `ZeroMQClient`
must supply its own timeout/supervision wrapper regardless of library choice
(already a stated invariant in component 01.3: "every call carries a
timeout"). Paired with `msgpax` for the MessagePack wire format
([ADR-0007](0007-msgpack-wire-format-for-zeromq-fallback.md#adr-0007)) —
`msgpax` decodes map keys as strings, not atoms, so `ZeroMQClient`'s response
handling pattern-matches on string keys (`"ok"`/`"error"`), not atoms.
