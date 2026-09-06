# Qwen tokenizer audit

The generation experiments already use vLLM's returned prompt_tokens, completion_tokens and total_tokens. No external token-counting service was used. The old codec-only measurements used the local tiktoken library with cl100k_base.

This follow-up calls POST /tokenize on the same Qwen deployment. Nine raw-string calls count the three blog examples in old TOON, current TOON and compact JSON, with add_special_tokens=false. Another 32 calls tokenize saved chat messages with add_generation_prompt=true, add_special_tokens=false and the original enable_thinking setting. All 32 counts exactly match the original chat-completion usage.prompt_tokens.

The blog's codec-only table now uses Qwen counts. Historical cl100k_base measurements remain unchanged and labeled in UPGRADE.md. No model generation was rerun, and the accuracy/token charts do not change.

requests.json and responses.json preserve payloads and responses including token IDs; summary.json records counts and matching results. The 32-chat check is a stratified sample covering suite/format/thinking combinations, not an independent retokenization of every request. Completion counts continue to come from the original generation responses; tokenizing visible answer text alone would omit reasoning and potentially generation control tokens.
