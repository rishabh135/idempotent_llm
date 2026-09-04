# Determinism cookbook — real run (2026-09-03T23:09:12.069337+00:00)

```text
determinism cookbook: 20 runs, n_predict=128, host=http://127.0.0.1:8080
request: temperature=0, top_k=1, top_p=1.0, seed=42, cache_prompt=false

  run   0: final_hash=a380b042a7c6d771  predicted_tok/s=294.4
  run   1: final_hash=a380b042a7c6d771  predicted_tok/s=287.8
  run   2: final_hash=a380b042a7c6d771  predicted_tok/s=285.4
  run   3: final_hash=a380b042a7c6d771  predicted_tok/s=287.8
  run   4: final_hash=a380b042a7c6d771  predicted_tok/s=279.6
  run   5: final_hash=a380b042a7c6d771  predicted_tok/s=293.0
  run   6: final_hash=a380b042a7c6d771  predicted_tok/s=282.4
  run   7: final_hash=a380b042a7c6d771  predicted_tok/s=299.1
  run   8: final_hash=a380b042a7c6d771  predicted_tok/s=282.6
  run   9: final_hash=a380b042a7c6d771  predicted_tok/s=298.0
  run  10: final_hash=a380b042a7c6d771  predicted_tok/s=287.7
  run  11: final_hash=a380b042a7c6d771  predicted_tok/s=292.4
  run  12: final_hash=a380b042a7c6d771  predicted_tok/s=282.2
  run  13: final_hash=a380b042a7c6d771  predicted_tok/s=258.0
  run  14: final_hash=a380b042a7c6d771  predicted_tok/s=297.0
  run  15: final_hash=a380b042a7c6d771  predicted_tok/s=291.7
  run  16: final_hash=a380b042a7c6d771  predicted_tok/s=291.7
  run  17: final_hash=a380b042a7c6d771  predicted_tok/s=284.2
  run  18: final_hash=a380b042a7c6d771  predicted_tok/s=280.3
  run  19: final_hash=a380b042a7c6d771  predicted_tok/s=289.7

PASS: 20/20 runs byte-identical (final hash a380b042a7c6d771)

cross-check against this repo's own int8 pipeline (scripts/demo.py --config int8-cpu-b1 --steps 64, a quick sanity contrast, NOT the full 512-step suite):

pipeline                                    runs  identical?  guarantee class
llama.cpp GGUF (this lab)                     20         yes  request-level, THIS machine + build only
DetLLM int8 (scripts/demo.py)                  1   n/a (ref)  request-level AND cross-hardware/cross-batch/cross-backend
    DetLLM int8-cpu-b1 hash: a4b8ced00476d16a

SCOPE LIMIT: this demonstrates single-machine, single-build 'request-level idempotency' only (deep-dive/README.md §1.1) — not cross-hardware/cross-batch bit-exactness. See --explain for the full argument.
```
