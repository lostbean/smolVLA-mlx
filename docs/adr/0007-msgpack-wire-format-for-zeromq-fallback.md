# The ZeroMQ fallback's wire format is MessagePack over REQ/REP, not JSON

<a id="adr-0007"></a>

The `infer_action` port's ZeroMQ fallback adapter needed a concrete
request/response wire format; the design left this unspecified, since it is
an implementation contract, not intent. We considered plain JSON, since it is
the simplest to inspect and debug, but rejected it as the default: an
`infer_action` request carries a real camera image (a dense array, easily
hundreds of KB) and a response carries a float action-chunk array — both
would need base64 encoding under JSON, inflating payload size by roughly a
third and adding encode/decode overhead on every tick-triggered call, on a
link that ADR-0003 already treats as a real resilience boundary worth
tightening, not loosening. We chose MessagePack instead: it carries binary
array data natively (no base64 tax), is a mature, well-supported library on
both sides (`msgpack-python` and Elixir's `msgpack` package), and keeps the
wire format debuggable with standard tooling (unlike a schema-first format
like Protobuf, which we also considered and rejected as unnecessary
ceremony for a single, stable request/response shape with no versioning
pressure yet). The socket pattern is ZeroMQ `REQ`/`REP` — synchronous
request/response, matching the `infer_action` port's own synchronous
contract; no pub/sub or streaming shape is introduced. The exact message
schema (field names, tensor encoding) is captured in
[model-runtime](../design/model-runtime/design.md) component 01.1 and
[control-loop](../design/control-loop/design.md) component 01.3.
