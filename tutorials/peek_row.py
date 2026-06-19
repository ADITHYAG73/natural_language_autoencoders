"""Peek at one AV-SFT row: the real (activation -> Claude explanation) training pair.
Usage: python peek_row.py /path/to/av_sft.parquet
"""
import sys
import numpy as np
import pyarrow.parquet as pq

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/nla_gemma4_26b_10doc/av_sft.parquet"
rows = pq.read_table(path).to_pylist()
r = rows[0]
v = np.asarray(r["activation_vector"], dtype=np.float32)

print("=== ONE AV-SFT ROW (Gemma-4 26B-A4B, layer 20) ===")
print("doc_id          :", r.get("doc_id"))
print("activation_layer:", r.get("activation_layer"), "| n_raw_tokens:", r.get("n_raw_tokens"))
print("activation_vector: dim", v.shape[0], "| L2 norm", round(float(np.linalg.norm(v)), 1))
print("\n--- PROMPT (the AV input; note the ㈜ injection slot) ---")
print(r["prompt"])
print("\n--- RESPONSE (Claude's explanation = the SFT target the AV learns to produce) ---")
print(r.get("response"))
src = r.get("detokenized_text_truncated")
if src:
    print("\n--- SOURCE TEXT the activation came from (what Claude actually described) ---")
    print(repr(src[:600]))
