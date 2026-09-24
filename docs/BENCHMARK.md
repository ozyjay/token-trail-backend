# Local CPU benchmark

Run on 24 September 2026 on a local arm64 Mac with Python 3.12.13, PyTorch 2.14.0 and Transformers 4.55.4. This local run used the pinned Qwen2.5-0.5B snapshot, float32 CPU inference, two OpenMP threads, a warm model cache and the fixed prompt `The moon is`. It is one run, so timings are indicative rather than a service-level target. The Docker image pins PyTorch 2.8.0; compare again inside the actual Space.

| Measure | Result |
| --- | ---: |
| Tokeniser load | 128.1 ms |
| Model load | 511.1 ms |
| Prompt tokenisation | 0.8 ms |
| First generation step | 66.8 ms |
| Later steps | 16.5–22.2 ms each |
| Total trace | 189.5 ms |
| Generated tokens | 8 |
| Peak process memory | 3290.5 MB |

The per-step measurements were 65.9, 22.2, 16.8, 16.5, 16.6, 16.6, 16.7 and 16.8 ms. This excludes network latency and cold model download. A Space may be substantially slower because CPU Basic has fewer cores, different hardware and cold starts.

Reproduce with `python3 scripts/benchmark.py` after downloading the pinned snapshot; set `HF_HOME` to the desired cache and `OMP_NUM_THREADS=2` for a closer comparison.
