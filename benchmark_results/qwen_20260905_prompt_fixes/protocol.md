# Prompt-contract correction follow-up

400 calls: the same 100 frozen cases, TOON and BAML-inspired adapters, native thinking off/on. Same model, temperature, top_p, per-case seed, 8,192-token cap, and concurrency 50. Baseline case and payload identities are retained; only adapter-rendered messages can change. No golds, cases, scoring rules, response parsers, or retries change.

TOON now preserves descriptions inside nested array schemas. The BAML-inspired adapter now requests one JSON object with the output wrapper its inherited JSON parser expects; branch-local recursion tracking also replaces the shared visited set. This remains the repository's BAML-inspired adapter, not full BoundaryML BAML.

These generic defects were diagnosed on the initial benchmark, so this follow-up uses development data, not an unseen validation set. Preserve both phases and do not present improvement as independently validated generalization. Fresh single samples and batching can change unchanged prompts' answers; before/after differences are not perfectly deterministic causal estimates. JSON/Chat references come from the separate initial 50-concurrency batch. Latency and token comparisons across phases are descriptive.

Data, source, licenses, original protocol and official scorers are preserved in ../qwen_20260905/. Full requests and source hashes for this follow-up are local. No additional tests were added.
