# Quantization report — `qwen3-0.6b-int8`

Full, unabridged output of [`scripts/quant_report.py`](../scripts/quant_report.py)
against the published artifact (`hf download
nathanbarry/detllm-qwen3-0.6b-int8`, `sha256(model.safetensors):
6658cea4dd89c613…`). The script reads only the safetensors *header* and
`config.json` — no weights are loaded, nothing here required a model run —
and reports, per transformer layer and in aggregate: parameter counts,
on-disk bytes, the exact `(data, m, k)` dyadic quantization scheme per
tensor family (see [`dyadic.py`](../src/detllm/dyadic.py)), a dtype
breakdown, and the size ratio against an equivalent fp16/fp32 checkpoint.

The headline numbers and the two non-obvious findings they imply (the LM
head isn't tied on disk; the RoPE table outweighs all quantization scales
combined) are summarized in the main [README](../README.md#quantization-report-exactly-whats-on-disk-layer-by-layer).
This file is the full raw report those numbers were drawn from.

Regenerate against your own artifact with:

```bash
uv run python scripts/quant_report.py            # this report
uv run python scripts/quant_report.py --full     # + one row per raw tensor (~760 rows)
make quant-report                                 # same as the first, via Makefile
```

```text
artifact: artifacts/qwen3-0.6b-int8
model: Qwen/Qwen3-0.6B  |  sha256(model.safetensors): 6658cea4dd89c613…
architecture: 28L  hidden=1024  heads=16/8 (Q/KV, GQA)  head_dim=128  intermediate=3072  vocab=151936  weight_bits=8

Dyadic-scale convention (see dyadic.py): every quantized tensor is
carried as (data, m, k) with real value ≈ data · m / 2^k. `data` is
the narrow integer stored below; `m` (mantissa) and `k` (binary
shift) are either a tensor alongside it or a scalar in config.json.

==============================================================================
SUMMARY
==============================================================================
logical weight parameters (this artifact)           751,566,848
  of which embed.e8 + head.w8t (untied on disk)     311,164,928
original checkpoint parameters (tied, bf16)         596,049,920
---------------------------------------------------------------
artifact size on disk (all tensors, incl. scales)      745.10 MB
  equivalent fp16 checkpoint size                       1.11 GB
  equivalent fp32 checkpoint size                       2.22 GB
this artifact is smaller than fp16 by                    1.526x
this artifact is smaller than fp32 by                    3.052x
effective bits / logical weight parameter                 8.32b  (8b weight + 0.32b scale/bookkeeping overhead)

------------------------------------------------------------------------------
BY STORAGE DTYPE
------------------------------------------------------------------------------
dtype      tensors        elements         bytes   % of size
I16              2      10,485,760      20.00 MB        2.7%
I64            563       1,094,656       8.35 MB        1.1%
I8             198     751,566,848     716.75 MB       96.2%

------------------------------------------------------------------------------
BY COMPONENT (summed across all layers)
------------------------------------------------------------------------------
component                 dtype  shape/layer       x28   total elems       bytes
Q-proj weight             I8     [1024, 2048]       28    58,720,256    56.00 MB
Q-proj scale              I64    [2048]             28        57,344   448.00 KB
K-proj weight             I8     [1024, 1024]       28    29,360,128    28.00 MB
K-proj scale              I64    [1024]             28        28,672   224.00 KB
V-proj weight             I8     [1024, 1024]       28    29,360,128    28.00 MB
V-proj scale              I64    [1024]             28        28,672   224.00 KB
O-proj weight             I8     [2048, 1024]       28    58,720,256    56.00 MB
O-proj scale              I64    [1024]             28        28,672   224.00 KB
Gate-proj weight          I8     [1024, 3072]       28    88,080,384    84.00 MB
Gate-proj scale           I64    [3072]             28        86,016   672.00 KB
Up-proj weight            I8     [1024, 3072]       28    88,080,384    84.00 MB
Up-proj scale             I64    [3072]             28        86,016   672.00 KB
Down-proj weight          I8     [3072, 1024]       28    88,080,384    84.00 MB
Down-proj scale           I64    [1024]             28        28,672   224.00 KB
Q-Norm gamma              I64    [128]              28         3,584    28.00 KB
K-Norm gamma              I64    [128]              28         3,584    28.00 KB
QK-smoothing factor       I64    [8, 128]           28        28,672   224.00 KB
attn-input smoothing      I64    [1024]             28        28,672   224.00 KB
mlp-input smoothing       I64    [1024]             28        28,672   224.00 KB
down-input smoothing      I64    [3072]             28        86,016   672.00 KB
o-input smoothing         I64    [2048]             28        57,344   448.00 KB
K-cache scale mantissa    I64    [8, 128]           28        28,672   224.00 KB
K-cache scale shift       I64    [8, 128]           28        28,672   224.00 KB
K-score scale mantissa    I64    [8]                28           224     1.75 KB
K-score scale shift       I64    [8]                28           224     1.75 KB
V-cache scale mantissa    I64    [8]                28           224     1.75 KB
V-cache scale shift       I64    [8]                28           224     1.75 KB
embed.e8                  I8     [151936, 1024]      1   155,582,464   148.38 MB
embed.m                   I64    [151936]            1       151,936     1.16 MB
embed.k                   I64    [151936]            1       151,936     1.16 MB
head.w8t                  I8     [1024, 151936]      1   155,582,464   148.38 MB
head.m                    I64    [151936]            1       151,936     1.16 MB
rope.cos                  I16    [40960, 128]        1     5,242,880    10.00 MB
rope.sin                  I16    [40960, 128]        1     5,242,880    10.00 MB

Legend (scheme per component):
  - Q-proj weight-style: int8, symmetric, per-output-channel scale (row of .m mantissa / 2^k, one shared k per tensor)
  - Q-proj scale-style: int64 per-output-channel mantissa (pairs with the tensor-wide shift k recorded in config.json)
  - Q-Norm gamma-style: int64 per-channel dyadic RMSNorm affine (gamma folded out of the main path at prep time)
  - QK-smoothing factor-style: int64 per-channel dyadic multiplier (analytic SmoothQuant-style activation smoothing, folded into the consumer weight at prep time)
  - K-cache scale mantissa-style: int64 static per-(KV-head[,channel]) dyadic K/V cache scale (calibrated once offline)
  - Token embedding-style: int8, symmetric, per-ROW (per-token-id) scale — each row has its OWN shift k too (embed.k), unlike weight matrices
  - RoPE lookup table-style: int16, global fixed-point cos/sin table, single shared shift k=rope_frac_bits (see config.json)

==============================================================================
PER-LAYER BREAKDOWN (the answer to 'how much does each layer cost')
==============================================================================
  L   attn params   mlp params  total params       bytes  b/param  worst attn err  worst mlp err
  0     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.050%         0.008%
  1     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.027%         0.012%
  2     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.085%         0.152%
  3     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.031%         0.008%
  4     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.047%         0.007%
  5     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.014%         0.007%
  6     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.037%         0.006%
  7     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.033%         0.017%
  8     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.019%         0.011%
  9     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.029%         0.011%
 10     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.038%         0.013%
 11     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.058%         0.010%
 12     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.017%         0.010%
 13     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.024%         0.010%
 14     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.050%         0.016%
 15     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.082%         0.010%
 16     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.050%         0.010%
 17     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.024%         0.008%
 18     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.049%         0.008%
 19     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.079%         0.008%
 20     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.036%         0.007%
 21     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.161%         0.011%
 22     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.050%         0.015%
 23     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.031%         0.016%
 24     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.033%         0.018%
 25     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.035%         0.012%
 26     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.052%         0.011%
 27     6,291,456    9,437,184    15,728,640    15.17 MB    8.09b          0.011%         0.072%
------------------------------------------------------------------------------------------------
'worst attn/mlp err' = 0.5/min_m, the worst-case relative per-channel weight-scale
error observed at prep time for that layer's linears (prepare.py asserts min_m >= 2^8,
i.e. <= 0.2%, for every channel of every weight in the model).

non-layer tensors (embed+head+rope)                311,164,928 params   320.23 MB
```
