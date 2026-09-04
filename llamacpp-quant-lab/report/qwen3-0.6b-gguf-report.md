gguf file: llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_M.gguf
sha256: c974d814f88ecb3e…
general.architecture: qwen3  general.file_type: 15  general.quantization_version: 2
tensors: 310  layers: 28  total elements: 596,049,920  total bytes (tensors): 372.65 MB  file size: 378.33 MB

==============================================================================
KEY-VALUE PAIRS
==============================================================================
  GGUF.version                                    : 3
  GGUF.tensor_count                               : 310
  GGUF.kv_count                                   : 30
  general.architecture                            : qwen3
  general.type                                    : model
  general.sampling.top_k                          : 20
  general.sampling.top_p                          : 0.949999988079071
  general.sampling.temp                           : 0.6000000238418579
  general.name                                    : Qwen3 0.6b Hf
  general.finetune                                : hf
  general.basename                                : qwen3
  general.size_label                              : 0.6B
  qwen3.block_count                               : 28
  qwen3.context_length                            : 40960
  qwen3.embedding_length                          : 1024
  qwen3.feed_forward_length                       : 3072
  qwen3.attention.head_count                      : 16
  qwen3.attention.head_count_kv                   : 8
  qwen3.rope.freq_base                            : 1000000.0
  qwen3.attention.layer_norm_rms_epsilon          : 9.999999974752427e-07
  qwen3.attention.key_length                      : 128
  qwen3.attention.value_length                    : 128
  tokenizer.ggml.model                            : gpt2
  tokenizer.ggml.pre                              : qwen2
  tokenizer.ggml.tokens                           : [151936 items]
  tokenizer.ggml.token_type                       : [151936 items]
  tokenizer.ggml.merges                           : [151387 items]
  tokenizer.ggml.eos_token_id                     : 151645
  tokenizer.ggml.padding_token_id                 : 151643
  tokenizer.ggml.bos_token_id                     : 151643
  tokenizer.chat_template                         : "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role =="… [4168 chars]
  general.quantization_version                    : 2
  general.file_type                               : 15

------------------------------------------------------------------------------
BY COMPONENT (summed across all layers; a family with mixed types
across layers gets one row per (family, type) pair)
------------------------------------------------------------------------------
component           dtype     x N     total elems       bytes
attn-norm gamma     F32        28          28,672   112.00 KB
Q-proj weight       Q4_K       28      58,720,256    31.50 MB
K-proj weight       Q4_K       28      29,360,128    15.75 MB
V-proj weight       Q4_K       14      14,680,064     7.88 MB
V-proj weight       Q6_K       14      14,680,064    11.48 MB
O-proj weight       Q4_K       28      58,720,256    31.50 MB
Q-Norm gamma        F32        28           3,584    14.00 KB
K-Norm gamma        F32        28           3,584    14.00 KB
ffn-norm gamma      F32        28          28,672   112.00 KB
Gate-proj weight    Q4_K       28      88,080,384    47.25 MB
Up-proj weight      Q4_K       28      88,080,384    47.25 MB
Down-proj weight    Q4_K       14      44,040,192    23.62 MB
Down-proj weight    Q6_K       14      44,040,192    34.45 MB
output_norm.weight  F32         1           1,024     4.00 KB
token_embd.weight   Q6_K        1     155,582,464   121.71 MB

------------------------------------------------------------------------------
PER-LAYER QUANTIZATION TYPE (0-27)
------------------------------------------------------------------------------
  L attn-norm   Q-proj wei  K-proj wei  V-proj wei  O-proj wei  Q-Norm gam  K-Norm gam  ffn-norm g  Gate-proj   Up-proj we  Down-proj   
  0 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
  1 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
  2 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
  3 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
  4 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
  5 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
  6 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
  7 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
  8 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
  9 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 10 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 11 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 12 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 13 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 14 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 15 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 16 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 17 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 18 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 19 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 20 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 21 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 22 F32         Q4_K        Q4_K        Q4_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q4_K        
 23 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 24 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 25 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 26 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        
 27 F32         Q4_K        Q4_K        Q6_K        Q4_K        F32         F32         F32         Q4_K        Q4_K        Q6_K        

==============================================================================
SUMMARY
==============================================================================
total weight elements                                596,049,920
total bytes (tensors)                                  372.65 MB
effective bits / weight (real, computed)                   5.24b
equivalent F16 size (same shapes)                        1.11 GB
this file is smaller than F16 by                          3.051x

------------------------------------------------------------------------------
BY GGML QUANT TYPE (whole file)
------------------------------------------------------------------------------
type         tensors        elements         bytes   % of size
F32              113          65,536     256.00 KB        0.1%
Q4_K             168     381,681,664     204.75 MB       54.9%
Q6_K              29     214,302,720     167.65 MB       45.0%
