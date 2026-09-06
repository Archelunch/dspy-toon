# BBEH reasoning-budget sensitivity

80 requests: all 20 selected BBEH mini cases, all four adapters, native thinking enabled. Use the corrected TOON/BAML prompts, identical case seeds/temperature/top_p, concurrency 50, and a 32,768-token completion cap. This is a separately labeled follow-up motivated by widespread 8,192-token truncation. Cases are unchanged; no selection based on which adapter passed. No retries. It is not an independent held-out validation set or a thinking-off comparison at 32K. The original 8K results remain intact.

Reference the corrected 8K run for TOON/BAML and original 8K run for JSON/Chat. Separate batches, single stochastic samples, caching and shared server load limit causal claims. Token usage includes native reasoning. Store truncations and HTTP/timeouts as failures in denominators. Source/scorer/data provenance is in the manifests and ../qwen_20260905/.
