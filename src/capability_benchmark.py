"""
Capability benchmark (IBM SkillsBuild: "The Intelligence Behind AI" --
speed of computation, scalability, real-time adjustment).

Measures inference latency for each trained model across increasing batch
sizes, by resampling the test set with replacement. This is a real,
measured comparison -- not a claim from memory -- so the numbers in the
report are whatever this script actually prints.
"""
import time
import numpy as np
import pandas as pd


def benchmark_inference(models: dict, X_test, batch_sizes=(1, 50, 500, 5000)) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for name, info in models.items():
        pipe = info["fitted_model"]
        for n in batch_sizes:
            idx = rng.integers(0, len(X_test), size=n)
            batch = X_test.iloc[idx]
            start = time.perf_counter()
            pipe.predict(batch)
            elapsed = time.perf_counter() - start
            rows.append({
                "model": name,
                "batch_size": n,
                "total_ms": round(elapsed * 1000, 3),
                "ms_per_sample": round((elapsed * 1000) / n, 5),
            })
    return pd.DataFrame(rows)
