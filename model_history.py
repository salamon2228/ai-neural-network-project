"""
Model history & benchmark storage.

Tracks every training run and evaluation per model in a persistent JSON file.
Enables the autopilot to prove improvement across runs and compare model versions.

Schema (memory/model_history.json):
{
  "models": {
    "<model_name>": {
      "created_at": "iso-timestamp",
      "runs": [
        {
          "run_id": "run_<n>",
          "timestamp": "iso",
          "type": "create" | "train" | "continue",
          "config": {...},          // model config if create/train
          "datasets": [...],        // filenames attached
          "training": {             // populated if training happened
            "iterations": 5000,
            "final_loss": 2.3,
            "final_reward": 0.6,
            "duration_sec": 1500
          },
          "benchmark": {            // populated when eval ran
            "benchmark_id": "bench_<n>",
            "overall_score": 0.72,
            "scores": {"coherence": 0.8, ...},
            "verdict": "GOOD",
            "samples": [...]
          },
          "notes": "free-form text"
        }
      ],
      "benchmarks": {
        "<benchmark_id>": {
          "created_at": "iso",
          "goal": "Russian literary text generation",
          "prompts": ["В этот день", ...],
          "language": "ru",
          "rubric": "free-form description"
        }
      }
    }
  }
}
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class ModelHistory:
    """Thread-safe JSON-backed history store for models, runs, and benchmarks."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"models": {}}

    def _save(self):
        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.storage_path)

    def _ensure_model(self, model_name: str) -> dict:
        models = self._data.setdefault("models", {})
        if model_name not in models:
            models[model_name] = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "runs": [],
                "benchmarks": {}
            }
        return models[model_name]

    def _next_id(self, prefix: str, collection) -> str:
        """Generate next sequential id like 'run_1', 'run_2', ..."""
        if isinstance(collection, list):
            nums = []
            for item in collection:
                rid = item.get("run_id", "")
                if rid.startswith(prefix + "_"):
                    try:
                        nums.append(int(rid.split("_")[-1]))
                    except ValueError:
                        pass
            return f"{prefix}_{(max(nums) + 1) if nums else 1}"
        else:
            nums = []
            for key in collection.keys():
                if key.startswith(prefix + "_"):
                    try:
                        nums.append(int(key.split("_")[-1]))
                    except ValueError:
                        pass
            return f"{prefix}_{(max(nums) + 1) if nums else 1}"

    # ─── Run tracking ────────────────────────────────

    def record_run(self, model_name: str, run_type: str,
                   config: Optional[dict] = None,
                   datasets: Optional[list] = None,
                   training: Optional[dict] = None,
                   notes: str = "") -> str:
        """Add a new run. Returns run_id."""
        with self._lock:
            m = self._ensure_model(model_name)
            run_id = self._next_id("run", m["runs"])
            m["runs"].append({
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "type": run_type,
                "config": config or {},
                "datasets": datasets or [],
                "training": training or {},
                "benchmark": {},
                "notes": notes
            })
            self._save()
            return run_id

    def attach_benchmark_result(self, model_name: str, run_id: str,
                                benchmark_id: str, result: dict) -> bool:
        """Attach an evaluation result to a specific run."""
        with self._lock:
            m = self._ensure_model(model_name)
            for r in m["runs"]:
                if r["run_id"] == run_id:
                    r["benchmark"] = {
                        "benchmark_id": benchmark_id,
                        **result
                    }
                    self._save()
                    return True
            return False

    def record_benchmark_only(self, model_name: str, benchmark_id: str,
                              result: dict, notes: str = "") -> str:
        """Record a standalone benchmark run (no training — e.g. baseline)."""
        with self._lock:
            m = self._ensure_model(model_name)
            run_id = self._next_id("run", m["runs"])
            m["runs"].append({
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "type": "benchmark",
                "config": {},
                "datasets": [],
                "training": {},
                "benchmark": {"benchmark_id": benchmark_id, **result},
                "notes": notes
            })
            self._save()
            return run_id

    # ─── Benchmark definitions ────────────────────────

    def save_benchmark(self, model_name: str, goal: str, prompts: list,
                       language: str = "en", rubric: str = "") -> str:
        """Store a benchmark definition (LLM-designed eval set)."""
        with self._lock:
            m = self._ensure_model(model_name)
            benchmark_id = self._next_id("bench", m["benchmarks"])
            m["benchmarks"][benchmark_id] = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "goal": goal,
                "prompts": prompts,
                "language": language,
                "rubric": rubric
            }
            self._save()
            return benchmark_id

    def get_benchmark(self, model_name: str, benchmark_id: str) -> Optional[dict]:
        with self._lock:
            m = self._data.get("models", {}).get(model_name)
            if not m:
                return None
            return m.get("benchmarks", {}).get(benchmark_id)

    def latest_benchmark(self, model_name: str) -> Optional[tuple]:
        """Return (benchmark_id, definition) for the most recent benchmark, or None."""
        with self._lock:
            m = self._data.get("models", {}).get(model_name)
            if not m or not m.get("benchmarks"):
                return None
            # Most recent by created_at
            items = list(m["benchmarks"].items())
            items.sort(key=lambda kv: kv[1].get("created_at", ""), reverse=True)
            bid, bdef = items[0]
            return (bid, bdef)

    # ─── Querying ─────────────────────────────────────

    def get_history(self, model_name: str) -> dict:
        """Full history record for a model."""
        with self._lock:
            m = self._data.get("models", {}).get(model_name)
            if not m:
                return {"model_name": model_name, "exists": False, "runs": [], "benchmarks": {}}
            # Return a snapshot to avoid lock holding
            return {
                "model_name": model_name,
                "exists": True,
                "created_at": m.get("created_at"),
                "runs": list(m.get("runs", [])),
                "benchmarks": dict(m.get("benchmarks", {})),
                "run_count": len(m.get("runs", []))
            }

    def list_models(self) -> list:
        """Summary of all tracked models."""
        with self._lock:
            out = []
            for name, m in self._data.get("models", {}).items():
                runs = m.get("runs", [])
                last_bench = None
                for r in reversed(runs):
                    if r.get("benchmark", {}).get("overall_score") is not None:
                        last_bench = r["benchmark"]
                        break
                out.append({
                    "model_name": name,
                    "created_at": m.get("created_at"),
                    "run_count": len(runs),
                    "latest_score": last_bench.get("overall_score") if last_bench else None,
                    "latest_verdict": last_bench.get("verdict") if last_bench else None
                })
            return out

    def compare_runs(self, model_name: str, run_a: str = None, run_b: str = None) -> dict:
        """Compare two runs (by run_id) on benchmark scores. Defaults to last two with benchmark data."""
        with self._lock:
            m = self._data.get("models", {}).get(model_name)
            if not m:
                return {"error": f"Model '{model_name}' has no history."}
            runs = [r for r in m.get("runs", []) if r.get("benchmark", {}).get("overall_score") is not None]
            if len(runs) < 2 and (run_a is None or run_b is None):
                return {"error": f"Not enough benchmarked runs to compare (found {len(runs)}, need 2)."}

            if run_a and run_b:
                a = next((r for r in m.get("runs", []) if r["run_id"] == run_a), None)
                b = next((r for r in m.get("runs", []) if r["run_id"] == run_b), None)
                if not a or not b:
                    return {"error": f"One of the run_ids not found: {run_a}, {run_b}"}
            else:
                a, b = runs[-2], runs[-1]

            sa = a.get("benchmark", {})
            sb = b.get("benchmark", {})
            overall_a = sa.get("overall_score", 0)
            overall_b = sb.get("overall_score", 0)
            delta = overall_b - overall_a

            score_deltas = {}
            for key in set(sa.get("scores", {}).keys()) | set(sb.get("scores", {}).keys()):
                va = sa.get("scores", {}).get(key, 0)
                vb = sb.get("scores", {}).get(key, 0)
                score_deltas[key] = {"before": va, "after": vb, "delta": round(vb - va, 4)}

            return {
                "model_name": model_name,
                "run_a": {"run_id": a["run_id"], "timestamp": a["timestamp"],
                          "verdict": sa.get("verdict"), "overall": overall_a},
                "run_b": {"run_id": b["run_id"], "timestamp": b["timestamp"],
                          "verdict": sb.get("verdict"), "overall": overall_b},
                "overall_delta": round(delta, 4),
                "improved": delta > 0.02,  # > 2% improvement = meaningful
                "score_deltas": score_deltas,
                "improvement_pct": round(delta * 100, 2)
            }
