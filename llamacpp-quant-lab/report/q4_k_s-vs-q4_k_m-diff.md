A: llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_S.gguf  (sha256 95bd2ee787f10039…)
B: llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_M.gguf  (sha256 c974d814f88ecb3e…)

==============================================================================
KEY-VALUE PAIR DIFF
==============================================================================
  GGUF.kv_count                                   : 30
  GGUF.tensor_count                               : 310
  GGUF.version                                    : 3
  general.architecture                            : qwen3
  general.basename                                : qwen3
  general.file_type                               : A=14 | B=15
  general.finetune                                : hf
  general.name                                    : Qwen3 0.6b Hf
  general.quantization_version                    : 2
  general.sampling.temp                           : 0.6000000238418579
  general.sampling.top_k                          : 20
  general.sampling.top_p                          : 0.949999988079071
  general.size_label                              : 0.6B
  general.type                                    : model
  qwen3.attention.head_count                      : 16
  qwen3.attention.head_count_kv                   : 8
  qwen3.attention.key_length                      : 128
  qwen3.attention.layer_norm_rms_epsilon          : 9.999999974752427e-07
  qwen3.attention.value_length                    : 128
  qwen3.block_count                               : 28
  qwen3.context_length                            : 40960
  qwen3.embedding_length                          : 1024
  qwen3.feed_forward_length                       : 3072
  qwen3.rope.freq_base                            : 1000000.0
  tokenizer.chat_template                         : "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role =="… [4168 chars]
  tokenizer.ggml.bos_token_id                     : 151643
  tokenizer.ggml.eos_token_id                     : 151645
  tokenizer.ggml.merges                           : [151387 items]
  tokenizer.ggml.model                            : gpt2
  tokenizer.ggml.padding_token_id                 : 151643
  tokenizer.ggml.pre                              : qwen2
  tokenizer.ggml.token_type                       : [151936 items]
  tokenizer.ggml.tokens                           : [151936 items]

==============================================================================
PER-TENSOR DIFF  (legend: ⬜ F32  🟥 Q4_K  🟧 Q5_K  🟩 Q6_K)
==============================================================================
name                      shape               A size   A    B size   B      Δbytes  changed?
blk.0.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.0.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.0.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.0.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.0.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.0.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.0.attn_v.weight       [1024, 1024]     704.00 KB   🟧 840.00 KB   🟩    +139,264  <-- CHANGED
blk.0.ffn_down.weight     [3072, 1024]       2.06 MB   🟧   2.46 MB   🟩    +417,792  <-- CHANGED
blk.0.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.0.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.0.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.1.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.1.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.1.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.1.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.1.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.1.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.1.attn_v.weight       [1024, 1024]     704.00 KB   🟧 840.00 KB   🟩    +139,264  <-- CHANGED
blk.1.ffn_down.weight     [3072, 1024]       2.06 MB   🟧   2.46 MB   🟩    +417,792  <-- CHANGED
blk.1.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.1.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.1.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.10.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.10.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.10.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.10.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.10.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.10.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.10.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.10.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.10.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.10.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.10.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.11.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.11.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.11.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.11.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.11.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.11.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.11.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.11.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.11.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.11.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.11.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.12.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.12.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.12.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.12.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.12.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.12.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.12.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.12.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.12.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.12.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.12.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.13.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.13.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.13.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.13.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.13.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.13.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.13.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.13.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.13.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.13.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.13.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.14.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.14.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.14.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.14.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.14.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.14.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.14.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.14.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.14.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.14.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.14.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.15.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.15.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.15.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.15.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.15.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.15.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.15.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.15.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.15.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.15.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.15.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.16.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.16.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.16.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.16.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.16.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.16.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.16.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.16.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.16.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.16.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.16.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.17.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.17.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.17.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.17.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.17.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.17.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.17.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.17.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.17.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.17.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.17.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.18.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.18.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.18.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.18.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.18.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.18.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.18.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.18.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.18.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.18.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.18.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.19.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.19.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.19.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.19.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.19.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.19.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.19.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.19.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.19.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.19.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.19.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.2.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.2.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.2.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.2.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.2.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.2.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.2.attn_v.weight       [1024, 1024]     704.00 KB   🟧 840.00 KB   🟩    +139,264  <-- CHANGED
blk.2.ffn_down.weight     [3072, 1024]       2.06 MB   🟧   2.46 MB   🟩    +417,792  <-- CHANGED
blk.2.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.2.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.2.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.20.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.20.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.20.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.20.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.20.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.20.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.20.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.20.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.20.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.20.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.20.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.21.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.21.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.21.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.21.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.21.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.21.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.21.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.21.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.21.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.21.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.21.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.22.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.22.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.22.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.22.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.22.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.22.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.22.attn_v.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.22.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.22.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.22.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.22.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.23.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.23.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.23.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.23.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.23.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.23.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.23.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.23.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.23.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.23.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.23.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.24.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.24.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.24.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.24.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.24.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.24.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.24.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.24.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.24.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.24.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.24.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.25.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.25.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.25.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.25.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.25.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.25.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.25.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.25.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.25.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.25.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.25.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.26.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.26.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.26.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.26.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.26.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.26.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.26.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.26.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.26.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.26.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.26.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.27.attn_k.weight      [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.27.attn_k_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.27.attn_norm.weight   [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.27.attn_output.weight [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.27.attn_q.weight      [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.27.attn_q_norm.weight [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.27.attn_v.weight      [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.27.ffn_down.weight    [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.27.ffn_gate.weight    [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.27.ffn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.27.ffn_up.weight      [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.3.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.3.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.3.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.3.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.3.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.3.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.3.attn_v.weight       [1024, 1024]     704.00 KB   🟧 576.00 KB   🟥    -131,072  <-- CHANGED
blk.3.ffn_down.weight     [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.3.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.3.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.3.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.4.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.4.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.4.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.4.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.4.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.4.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.4.attn_v.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.4.ffn_down.weight     [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.4.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.4.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.4.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.5.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.5.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.5.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.5.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.5.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.5.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.5.attn_v.weight       [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.5.ffn_down.weight     [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.5.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.5.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.5.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.6.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.6.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.6.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.6.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.6.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.6.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.6.attn_v.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.6.ffn_down.weight     [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.6.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.6.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.6.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.7.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.7.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.7.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.7.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.7.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.7.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.7.attn_v.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.7.ffn_down.weight     [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.7.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.7.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.7.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.8.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.8.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.8.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.8.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.8.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.8.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.8.attn_v.weight       [1024, 1024]     576.00 KB   🟥 840.00 KB   🟩    +270,336  <-- CHANGED
blk.8.ffn_down.weight     [3072, 1024]       1.69 MB   🟥   2.46 MB   🟩    +811,008  <-- CHANGED
blk.8.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.8.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.8.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.9.attn_k.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.9.attn_k_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.9.attn_norm.weight    [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.9.attn_output.weight  [2048, 1024]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.9.attn_q.weight       [1024, 2048]       1.12 MB   🟥   1.12 MB   🟥          +0
blk.9.attn_q_norm.weight  [128]             512.00 B   ⬜  512.00 B   ⬜          +0
blk.9.attn_v.weight       [1024, 1024]     576.00 KB   🟥 576.00 KB   🟥          +0
blk.9.ffn_down.weight     [3072, 1024]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.9.ffn_gate.weight     [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
blk.9.ffn_norm.weight     [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
blk.9.ffn_up.weight       [1024, 3072]       1.69 MB   🟥   1.69 MB   🟥          +0
output_norm.weight        [1024]             4.00 KB   ⬜   4.00 KB   ⬜          +0
token_embd.weight         [1024, 151936]   121.71 MB   🟩 121.71 MB   🟩          +0

------------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------------
changed tensors                                 29 / 310
total size A                                       359.84 MB
total size B                                       372.65 MB
Δ size (B - A)                                   +13,434,880 bytes
bits/weight A                                          5.06b
bits/weight B                                          5.24b

ASSERTION PASSED: all 29 changed tensors are attn_v.weight/ffn_down.weight, as expected for a Q4_K_S-vs-Q4_K_M-style comparison.
