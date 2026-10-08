Device code from SageAttention (https://github.com/thu-ml/SageAttention, commit d1a57a5, csrc/*.cuh and
csrc/qattn/attn_utils.cuh), Apache-2.0 (LICENSE). Used by ../../fattn-sage.cu.

Changes: torch includes and the torch-only utils.cuh removed, include paths flattened, `compute_int_qk` (Q in shared
memory) takes `accumulate` to add to the scores instead of initializing them (Tilde's 192-dim Q/K run as 128 + 64).
