"""One-off: prove which run_code_benchmark_scoring is actually loaded in the eval container.

Usage (inside the eval image):
    python /opt/scripts/diagnose_assembly.py <benchmark-results-dir>
"""

import inspect
import sys

import polars as pl

from llm_fine_tune.evaluation import run_code_benchmark_scoring as s

print("MODULE FILE     :", s.__file__)
print("has _assemble_java:", hasattr(s, "_assemble_java"))
print("--- source of _assemble_multipl_e_program ---")
print(inspect.getsource(s._assemble_multipl_e_program))

if len(sys.argv) > 1:
    d = sys.argv[1]
    df = pl.read_parquet(f"{d}/code_benchmark_generations.parquet")
    print("LANGS in parquet:", sorted(set(df["language"].to_list())))
    row = df.filter(pl.col("config") == "humaneval-cpp").row(0, named=True)
    prog = s._assemble_multipl_e_program(
        row["completion"], row["tests"], row["language"], row["prompt"]
    )
    print("--- assembled cpp[0] (first 300 chars) ---")
    print(prog[:300])
