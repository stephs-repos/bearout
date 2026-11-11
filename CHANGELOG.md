# Changelog

## 0.0.4 (2026-11-11)

### Dependencies

- The `anthropic` extra now allows SDK 2.x: `anthropic>=0.40,<3` (was `>=0.40,<1.0`). Tested at
  0.40.0, 1.0.0, 2.0.0, 2.6.0 and 2.8.0.

### Changed

- `AnthropicChat` sends a forwarded `temperature` through `extra_body` and pins that request to API
  version `2023-06-01`. SDK 1.0 removed `temperature` from `messages.create()`, and SDK 2.x defaults
  to API version `2026-10-21`, which rejects sampling parameters with a 400. On SDK 0.x the request
  on the wire is unchanged.
- The measured configuration is unchanged: the same judge (`claude-sonnet-4-5-20250929`), temperature 0,
  prompts, and verdict logic. Requests without a temperature (any model other than the measured judge,
  unless `forward_temperature=True`) use the SDK's default API version.
- Retry and timeout behaviour is unchanged. The adapter still uses the SDK defaults (2 retries,
  600 s timeout, 5 s connect), which are identical from 0.40 through 2.8. A test now pins this.

### Notes

- SDK 1.11 and later emit a `DeprecationWarning` on every call to `claude-sonnet-4-5-20250929`, which
  the SDK lists as reaching end-of-life on 30 November 2026. bearout's published error rates were
  measured with that judge. When it is retired, the default configuration stops working and the rates
  need re-measuring against a successor.

## 0.0.1 – 0.0.3

Released before this changelog was started; see the `v0.0.1`–`v0.0.3` tags.
