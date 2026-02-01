# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-01

### Added

- Full TOON spec v3.0 compliance for "objects as list items" format
- Support for encoding objects with tabular arrays as first field
- Support for encoding objects with primitive arrays as first field
- Support for empty objects as list items (bare hyphen)
- New test suite for v3.0 format compliance (8 tests)

### Changed

- **Breaking**: Changed encoding of list items with tabular first field to v3.0 format
  - Old: `-\n  field[N]{f1,f2}:` (bare hyphen with field at depth+1)
  - New: `- field[N]{f1,f2}:` (tabular header on hyphen line)
- Updated system prompts to describe v3.0 format for LLMs
- Updated README with v3.0 format examples
- Improved SEO keywords in package metadata

### Fixed

- Fixed decoder to properly parse bare `-` as empty object in list items
- Fixed test expectation for primitive array schema output

## [0.2.1] - 2025-01-XX

### Added

- Initial beta release
- Core TOON encoding/decoding functionality
- DSPy ToonAdapter with streaming support
- Token usage benchmarks
- Async support via `dspy.asyncify()`

[0.3.0]: https://github.com/Archelunch/dspy-toon/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Archelunch/dspy-toon/releases/tag/v0.2.1
