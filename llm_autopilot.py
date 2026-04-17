"""
LLM Autopilot — интеллектуальный агент, который автоматически подбирает данные и обучает модель.
Поддерживает: Anthropic (Claude), OpenAI, OpenAI-совместимые API (ollama, LM Studio).
"""

import json
import time
import threading
import urllib.request
import urllib.parse
import ssl
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

# ============================
# SYSTEM PROMPT
# ============================

SYSTEM_PROMPT = """You are an expert AI engineer autopilot. You train small transformer language models and you operate like a real ML engineer: you form a hypothesis, you establish a baseline, you train, you measure, you PROVE improvement with numbers, and you remember past work across runs.

## ABSOLUTE RULES (violating these = failure):
1. NEVER invent or guess dataset names/IDs. ONLY use IDs returned by list_catalog or search_huggingface.
2. ALWAYS read the result of each tool call before proceeding. If a tool returns an error, STOP and handle it.
3. NEVER say "done" or report success if any step actually failed.
4. NEVER proceed to training if 0 datasets were successfully attached.
5. You MUST design a benchmark BEFORE training and MUST re-run it AFTER training — and PROVE improvement via compare_runs.
6. Check get_model_history BEFORE making decisions — if this model has been trained before, learn from those runs.
7. Do NOT rely only on list_catalog templates. If the goal is niche, use search_huggingface, inspect_dataset, and combine multiple sources.
8. If the user provided QUALITY TARGETS (target_score, min_improvement_pct, must_include, must_avoid) — these are NOT suggestions. You MUST call check_quality_targets after the post-training benchmark and MUST NOT report is_final=true while any numeric target fails. Honest failure is better than fake success.

## Your Process (follow EXACTLY):

### Phase 0: Understand the goal & check history
1. Analyze the user's goal: language, genre, quality expectations, time budget.
2. If a model name is implied or the user wants to continue an existing model, call get_model_history to see past training runs and benchmark scores. Factor this into your plan.
3. report_to_user: explain your plan (datasets to try, model size, benchmark strategy, expected runtime).

### Phase 1: Data Collection (active hunting, not templates)
4. Start with list_catalog for a quick win, BUT also use search_huggingface for goal-specific datasets when the catalog looks generic.
5. Download 3-6 candidate datasets. For each successful download, call inspect_dataset to verify it actually fits the goal (right language, reasonable size, relevant style). Skip datasets that look off-topic.
6. Keep a list of dataset filenames that you inspected and judged to fit the goal.

### Phase 2: Model Creation
7. Create the model using parameters based on the goal (see guide below). For Russian or complex English, bias toward larger vocab (15k-20k) and more layers.
8. Attach ONLY datasets that passed inspection. If 0 fit → STOP and re-do Phase 1 with different queries.

### Phase 3: Baseline benchmark (MANDATORY)
9. Call design_benchmark with 5-8 prompts tailored to the goal. Prompts must be diverse (simple + challenging, cover the target style). Write a short rubric describing what "good output" looks like.
10. Call run_benchmark(model_name, label="baseline"). A freshly created model will score terribly (random output) — that's expected; this is your reference point.

### Phase 4: Training
11. Call start_training with appropriate iterations. For any serious quality target: at least 5000 iterations; for Russian literary/creative: 8000-15000.
12. Training runs in the background. You will receive periodic progress updates and a completion notification. While training runs, you should stay quiet — monitoring is automatic.

### Phase 5: Post-training benchmark & proof of improvement (MANDATORY)
13. Call run_benchmark(model_name, label="after_train") with the SAME benchmark as the baseline.
14. Call compare_runs(model_name). This returns overall_delta, per-metric deltas, and an `improved` flag.
15. Generate 2-3 ad-hoc samples with generate_sample for qualitative spot-check (different prompts/temperatures).

### Phase 6: Check user's quality targets (MANDATORY if targets specified)
16. Call check_quality_targets(model_name). This returns per-criterion pass/fail for:
    - target_score (minimum overall_score the user accepts)
    - min_improvement_pct (minimum improvement vs baseline)
    - must_include / must_avoid (qualitative — YOU judge from samples)
17. If `all_numeric_checks_passed: false` → you CANNOT mark is_final=true. You MUST iterate.
18. For qualitative checks (must_include / must_avoid), examine the benchmark samples you already have. If the model outputs clearly violate must_avoid (e.g. pure gibberish, wrong language, excessive <UNK>) — treat it as failure and iterate.

### Phase 7: Decide — ship or iterate
19. Ship (report_to_user with is_final=true) ONLY IF:
    - compare_runs shows `improved: true` AND
    - check_quality_targets shows all numeric checks passed AND
    - the samples qualitatively satisfy must_include / must_avoid
20. Otherwise iterate (up to 2 retries total):
    - Diagnose root cause: loss too high (need more iterations)? vocab too small (<UNK> heavy)? wrong datasets (off-topic)? repetition (need more diverse data)?
    - Fix ONE thing at a time: continue training with more iterations, OR re-attach better datasets, OR adjust params and recreate.
    - Re-run benchmark, compare_runs, check_quality_targets.
21. If after 2 retries targets are STILL unmet → call report_to_user(is_final=true) with HONEST failure:
    - State which targets were not met and by how much
    - Show sample outputs
    - Explain what you tried and what you would recommend next (e.g. "need more data" / "need GPU for longer training" / "target score unrealistic for this dataset size")
    - NEVER pretend success when check_quality_targets failed.

## Model Parameter Guide
| Goal | layers | d_model | d_ff | vocab | iterations | lr |
|------|--------|---------|------|-------|------------|------|
| Quick test | 4 | 128 | 512 | 8000 | 1500-3000 | 3e-4 |
| Short creative | 4 | 256 | 1024 | 10000 | 4000-6000 | 3e-4 |
| Standard text | 6 | 256 | 1024 | 15000 | 6000-10000 | 3e-4 |
| Quality text | 8 | 384 | 1536 | 20000 | 10000-20000 | 1e-4 |
| Russian literary | 6-8 | 256-384 | 1024 | 15000-20000 | 8000-15000 | 3e-4 |
| Code | 6 | 384 | 1536 | 25000 | 6000-12000 | 3e-4 |

## Critical Details
- attach_dataset filename = "{catalog_id}.txt" (e.g., "ruslit_war_and_peace.txt")
- For Russian: vocab MUST be ≥ 15000 (rich morphology)
- More iterations = better quality but takes longer
- If user set a time limit, adjust iterations AND reserve ~5 min at the end for benchmarking + comparison
- batch_size: 16 default; 8 for large models
- A loss < 2.5 after training usually means decent quality
- A loss > 4.0 after many iterations usually means the model learned almost nothing useful — suspect bad data or wrong params

## Proof-of-improvement bar
Meaningful improvement = `overall_delta` > 0.02 (2%+ on the composite score). Smaller than that is noise — report it honestly as "no meaningful improvement" and describe what you would try next.

## Time Budget
{time_budget_info}
"""

# ============================
# TOOL DEFINITIONS
# ============================

AUTOPILOT_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "list_catalog",
            "description": "List available datasets in the catalog. Can filter by category and language. Returns dataset id, name, category, language, description, size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category: literature_en, literature_ru, detective, horror, scifi, adventure, fairy_tales, drama, poetry, mythology, philosophy, science, history, biography, spiritual, code, conversations, news. Optional."},
                    "language": {"type": "string", "description": "Filter by language: en, ru, multi. Optional."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_huggingface",
            "description": "Search HuggingFace for text datasets by query. Returns dataset names, descriptions, download counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_dataset",
            "description": "Download a dataset from the catalog by its ID. The dataset will be saved as {id}.txt in the books directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "The catalog dataset ID (e.g. 'gutenberg_sherlock')"}
                },
                "required": ["dataset_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_model",
            "description": "Create a new transformer language model with given parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Model name (no spaces, use underscores)"},
                    "vocab_size": {"type": "integer", "description": "Vocabulary size (default 15000)"},
                    "d_model": {"type": "integer", "description": "Model dimension (default 256)"},
                    "num_layers": {"type": "integer", "description": "Number of transformer layers (default 6)"},
                    "num_heads": {"type": "integer", "description": "Number of attention heads (default 8)"},
                    "d_ff": {"type": "integer", "description": "Feed-forward dimension (default 1024)"},
                    "max_seq_len": {"type": "integer", "description": "Max sequence length (default 256)"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "attach_dataset",
            "description": "Attach a downloaded dataset to a model for training.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model"},
                    "dataset_name": {"type": "string", "description": "Filename of the dataset (e.g. 'gutenberg_sherlock.txt')"}
                },
                "required": ["model_name", "dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_training",
            "description": "Start training a model. Training runs in the background. Use check_training_status to monitor progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model to train"},
                    "max_iterations": {"type": "integer", "description": "Maximum training iterations (default 5000)"},
                    "batch_size": {"type": "integer", "description": "Batch size (default 16)"},
                    "learning_rate": {"type": "number", "description": "Learning rate (default 0.0003)"}
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_training_status",
            "description": "Check current training progress. Returns is_training, current_iteration, max_iterations, current_loss, current_reward, eta_seconds.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sample",
            "description": "Generate a text sample from the trained model to evaluate quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model"},
                    "prompt": {"type": "string", "description": "Starting text prompt"},
                    "max_length": {"type": "integer", "description": "Maximum tokens to generate (default 100)"},
                    "temperature": {"type": "number", "description": "Sampling temperature (default 0.8)"}
                },
                "required": ["model_name", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": "Run automatic quality evaluation on a trained model. Generates multiple samples and scores them on coherence, diversity, language correctness. Returns a quality report with scores and verdict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model to evaluate"},
                    "language": {"type": "string", "description": "Expected language: 'ru' or 'en'"},
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of prompts to test (3-5 recommended). If empty, auto-generates prompts."
                    }
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "design_benchmark",
            "description": "Create a custom evaluation benchmark for a model based on the user's goal. YOU (the LLM) design the prompts — they should be specific to the goal (genre, language, style). This benchmark will be used to compare training runs. Call BEFORE training to establish a baseline, then run the SAME benchmark after training to prove improvement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model this benchmark is for"},
                    "goal": {"type": "string", "description": "What the model is supposed to do (e.g. 'Russian detective stories in Sherlock Holmes style')"},
                    "language": {"type": "string", "description": "'ru' or 'en'"},
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "5-8 diverse prompts that represent the model's target use-case. Mix simple and challenging. Each prompt should be 2-6 words — a natural beginning for the generated text."
                    },
                    "rubric": {"type": "string", "description": "Brief description of what 'good' output looks like for this goal. Used for later human review."}
                },
                "required": ["model_name", "goal", "prompts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_benchmark",
            "description": "Run the current benchmark on a model and record the result in history. Use this to establish a baseline BEFORE training (on a fresh or existing model), and then AGAIN after training to prove improvement. Automatically uses the most recent benchmark for this model. Returns overall_score, per-metric scores, verdict, and samples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model to benchmark"},
                    "label": {"type": "string", "description": "Short tag for this benchmark run, e.g. 'baseline', 'after 5k iter', 'final'"}
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_history",
            "description": "Retrieve the full training and benchmark history for a model. Returns all runs (creations, training sessions, benchmarks) with timestamps, configs, datasets used, and scores. Use this to understand what has already been tried before deciding next steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model"}
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_runs",
            "description": "Compare benchmark scores between two runs (before/after training) to PROVE whether quality improved. Returns overall delta, per-metric deltas, and a boolean 'improved' flag. By default compares the two most recent benchmarked runs. Use this after retraining to demonstrate improvement numerically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model"},
                    "run_a": {"type": "string", "description": "Optional: earlier run_id (default: second-to-last benchmarked run)"},
                    "run_b": {"type": "string", "description": "Optional: later run_id (default: last benchmarked run)"}
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": "Peek into a downloaded dataset file to understand its contents before attaching. Returns first 500 chars, line count, approx word count, and language detection. Helps decide which datasets actually fit the goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Dataset filename (e.g. 'ruslit_war_and_peace.txt')"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_quality_targets",
            "description": "Check whether the latest benchmarked state of a model MEETS the user's quality targets (target_score, min_improvement_pct, must_include, must_avoid). Returns per-criterion pass/fail. You MUST call this after the post-training benchmark and BEFORE reporting final results. If any numeric criterion fails, you MUST iterate (more training, better data, or honestly report failure). NEVER say the job is done if this returns failures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model to check"}
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_to_user",
            "description": "Send a message to the user. Use this to explain your reasoning, report progress, or deliver the final report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to show the user"},
                    "is_final": {"type": "boolean", "description": "Set to true for the final report (ends the autopilot session)"}
                },
                "required": ["message"]
            }
        }
    }
]

# Anthropic format
AUTOPILOT_TOOLS_ANTHROPIC = []
for _t in AUTOPILOT_TOOLS_OPENAI:
    _func = _t["function"]
    AUTOPILOT_TOOLS_ANTHROPIC.append({
        "name": _func["name"],
        "description": _func["description"],
        "input_schema": _func["parameters"]
    })


# ============================
# LLM PROVIDER
# ============================

class LLMProvider:
    """Unified interface for LLM API calls (Anthropic, OpenAI, OpenAI-compatible)."""

    def __init__(self, provider: str, api_key: str, endpoint: str = None, model: str = None):
        self.provider = provider  # "anthropic" | "openai" | "openai_compatible"
        self.api_key = api_key
        self.model = model
        self._log_fn = None  # Set by LLMAutopilot for status logging

        if provider == "anthropic":
            self.endpoint = "https://api.anthropic.com/v1/messages"
            self.model = model or "claude-sonnet-4-20250514"
        elif provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = model or "gpt-4o"
        elif provider == "openai_compatible":
            self.endpoint = (endpoint or "http://localhost:11434/v1") + "/chat/completions"
            if not self.endpoint.startswith("http"):
                self.endpoint = "http://" + self.endpoint
            # Ensure /chat/completions suffix
            if "/chat/completions" not in self.endpoint:
                self.endpoint = self.endpoint.rstrip("/") + "/chat/completions"
            self.model = model or "llama3"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _trim_messages(self, messages: list, max_messages: int = 12) -> list:
        """Keep system message + last N messages to avoid token overflow."""
        # Filter out any non-dict entries (safety check)
        messages = [m for m in messages if isinstance(m, dict)]
        if len(messages) <= max_messages + 1:
            return messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        # Keep last max_messages non-system messages
        trimmed = non_system[-max_messages:]
        # Aggressively truncate tool results to save tokens
        result = []
        for m in trimmed:
            content = m.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            elif not isinstance(content, str):
                content = str(content) if content else ""
            if m.get("role") == "tool" and len(content) > 300:
                m = dict(m)
                m["content"] = content[:300] + "...(truncated)"
            elif m.get("role") == "assistant" and len(content) > 500:
                m = dict(m)
                m["content"] = content[:500] + "...(truncated)"
            result.append(m)
        return system_msgs + result

    def chat(self, messages: list, tools: list = None) -> dict:
        """Send chat request. Returns {"content": str|None, "tool_calls": list|None, "stop_reason": str}"""
        if self.provider == "anthropic":
            return self._call_anthropic(messages, tools)
        else:
            return self._call_openai(messages, tools)

    def _call_openai(self, messages: list, tools: list = None) -> dict:
        """Call OpenAI or OpenAI-compatible API."""
        # Trim conversation history to avoid token overflow
        trimmed = self._trim_messages(messages)

        body = {
            "model": self.model,
            "messages": trimmed,
            "max_tokens": 1024,
        }
        if tools:
            body["tools"] = AUTOPILOT_TOOLS_OPENAI
            body["tool_choice"] = "auto"

        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AZR-Autopilot/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        ctx = ssl.create_default_context()

        import re as _re
        max_retries = 4
        timeout_sec = 45
        for _attempt in range(max_retries):
            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            try:
                if self._log_fn and _attempt == 0:
                    self._log_fn("status", f"Calling LLM ({self.model})...")
                resp = urllib.request.urlopen(req, timeout=timeout_sec, context=ctx)
                result = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                # HTTPError is a subclass of URLError — handle FIRST
                error_body = e.read().decode("utf-8", errors="ignore")
                # Rate limit — wait and retry (capped so user doesn't wait forever)
                if e.code == 429:
                    wait_match = _re.search(r'try again in (\d+\.?\d*)', error_body)
                    wait_time = float(wait_match.group(1)) + 3 if wait_match else 20
                    wait_time = max(wait_time, 10)
                    wait_time = min(wait_time, 45)
                    if _attempt < max_retries - 1:
                        if self._log_fn:
                            self._log_fn("status", f"Rate limit (429), waiting {int(wait_time)}s ({_attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    raise Exception(f"LLM API rate limit: exceeded after {max_retries} retries. Try a different provider (Google AI Studio / Ollama Cloud).")
                # 404 = wrong endpoint or model name — fail fast
                if e.code == 404:
                    raise Exception(f"LLM API error 404: model or endpoint not found. URL: {self.endpoint}, model: {self.model}. Error: {error_body[:300]}")
                # 401/403 = bad API key — fail fast
                if e.code in (401, 403):
                    raise Exception(f"LLM API auth error {e.code}: check your API key. Error: {error_body[:300]}")
                # Groq/Llama: tool_use_failed with failed_generation
                parsed = self._parse_failed_tool_call(error_body)
                if parsed:
                    return parsed
                # 5xx — retryable
                if 500 <= e.code < 600 and _attempt < max_retries - 1:
                    if self._log_fn:
                        self._log_fn("status", f"Server error {e.code}, retrying ({_attempt+1}/{max_retries})...")
                    time.sleep(3)
                    continue
                raise Exception(f"LLM API error {e.code}: {error_body[:500]}")
            except urllib.error.URLError as e:
                # Timeout or connection error
                if _attempt < max_retries - 1:
                    if self._log_fn:
                        self._log_fn("status", f"API timeout/conn error, retrying ({_attempt+1}/{max_retries}): {str(e)[:80]}")
                    time.sleep(3)
                    continue
                raise Exception(f"LLM API connection failed after {max_retries} attempts: {e}")
        else:
            raise Exception("LLM API: too many retries")

        # Debug: log raw response structure to help diagnose issues
        try:
            return self._parse_openai_response(result)
        except Exception as parse_err:
            # Log the raw response for debugging
            raw_dump = json.dumps(result, ensure_ascii=False, default=str)[:1000]
            raise Exception(f"Failed to parse LLM response: {parse_err}\nRaw response: {raw_dump}")

    def _parse_openai_response(self, result: dict) -> dict:
        """Parse OpenAI-compatible API response. Handles Gemini quirks."""
        if not isinstance(result, dict):
            raise Exception(f"Expected dict response, got {type(result).__name__}: {str(result)[:200]}")

        choices = result.get("choices")
        if not choices or not isinstance(choices, list):
            raise Exception(f"No 'choices' in response. Keys: {list(result.keys())}")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise Exception(f"Choice is {type(choice).__name__}, not dict: {str(choice)[:200]}")

        msg = choice.get("message")
        if not isinstance(msg, dict):
            # Some APIs put message content directly
            msg = {"role": "assistant", "content": str(msg) if msg else ""}

        # Normalize content: Gemini may return content as a list of parts
        raw_content = msg.get("content")
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", str(part)))
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            content_text = "".join(text_parts) or None
        else:
            content_text = raw_content

        tool_calls = None
        raw_tool_calls = msg.get("tool_calls")
        if raw_tool_calls and isinstance(raw_tool_calls, list):
            tool_calls = []
            for tc in raw_tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function")
                if not isinstance(func, dict):
                    continue
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                elif not isinstance(args, dict):
                    args = {}
                tool_calls.append({
                    "id": tc.get("id", f"call_{id(tc)}"),
                    "name": func.get("name", "unknown"),
                    "arguments": args
                })
            if not tool_calls:
                tool_calls = None

        # Fallback: parse tool calls from content if model uses XML-like format
        if not tool_calls and content_text:
            parsed = self._parse_failed_tool_call(json.dumps({"error": {"failed_generation": content_text}}))
            if parsed and parsed.get("tool_calls"):
                return parsed

        # Build a clean message for conversation history (content must be string or None)
        clean_msg = {"role": "assistant", "content": content_text or ""}
        # Preserve tool_calls in history if present (OpenAI format requires it)
        if tool_calls:
            clean_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                for tc in tool_calls
            ]

        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "stop_reason": choice.get("finish_reason", "stop"),
            "_raw_message": clean_msg
        }

    def _parse_failed_tool_call(self, error_body: str) -> dict:
        """Parse Groq/Llama failed_generation format: <function=name>{json}</function>"""
        try:
            err = json.loads(error_body)
            if not isinstance(err, dict):
                return None
            error_obj = err.get("error", {})
            if not isinstance(error_obj, dict):
                return None
            failed = error_obj.get("failed_generation", "")
            if not failed:
                return None
            # Parse various Llama formats:
            # <function=name>{"arg": "val"}</function>
            # <function=name,{"arg": "val"}</function>
            # <function=name>{$"arg": "val"}</function>
            # <function=name[]{"arg": "val"}</function>
            # <function=name() {"arg": "val"}</function>
            import re
            match = re.search(r'<function=(\w+)[^{]*(\{.*\})\s*</function>', failed, re.DOTALL)
            if match:
                func_name = match.group(1)
                try:
                    args = json.loads(match.group(2).replace("$", ""))
                except json.JSONDecodeError:
                    args = {}
                tc_id = f"call_{func_name}_{id(failed)}"
                return {
                    "content": None,
                    "tool_calls": [{
                        "id": tc_id,
                        "name": func_name,
                        "arguments": args
                    }],
                    "stop_reason": "tool_calls",
                    "_raw_message": {"role": "assistant", "content": None, "tool_calls": [{
                        "id": tc_id, "type": "function",
                        "function": {"name": func_name, "arguments": json.dumps(args)}
                    }]}
                }
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _call_anthropic(self, messages: list, tools: list = None) -> dict:
        """Call Anthropic Messages API."""
        # Separate system message
        system_text = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "tool":
                # Anthropic expects tool results inside user messages
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", "unknown"),
                        "content": m["content"]
                    }]
                })
            else:
                api_messages.append(m)

        body = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system_text:
            body["system"] = system_text
        if tools:
            body["tools"] = AUTOPILOT_TOOLS_ANTHROPIC

        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AZR-Autopilot/1.0",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        ctx = ssl.create_default_context()

        resp = urllib.request.urlopen(req, timeout=120, context=ctx)
        result = json.loads(resp.read().decode("utf-8"))

        content_text = ""
        tool_calls = []

        for block in result.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block.get("input", {})
                })

        return {
            "content": content_text or None,
            "tool_calls": tool_calls or None,
            "stop_reason": result.get("stop_reason", "end_turn"),
            "_raw_content": result.get("content", [])
        }

    def build_assistant_message(self, response: dict) -> dict:
        """Build the assistant message to add to conversation history."""
        if self.provider == "anthropic":
            return {"role": "assistant", "content": response.get("_raw_content", [])}
        else:
            msg = response.get("_raw_message", {"role": "assistant", "content": response.get("content", "")})
            # Ensure message is a dict with string content (not list) for OpenAI-compatible APIs
            if not isinstance(msg, dict):
                msg = {"role": "assistant", "content": str(msg) if msg else ""}
            if isinstance(msg.get("content"), list):
                parts = []
                for p in msg["content"]:
                    if isinstance(p, dict):
                        parts.append(p.get("text", ""))
                    elif isinstance(p, str):
                        parts.append(p)
                msg = dict(msg)
                msg["content"] = "".join(parts)
            return msg

    def build_tool_result_message(self, tool_call_id: str, result_str: str, tool_name: str = "") -> dict:
        """Build tool result message for conversation history."""
        msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_str
        }
        # Gemini requires 'name' field in tool responses
        if tool_name:
            msg["name"] = tool_name
        return msg


# ============================
# TOOL EXECUTOR
# ============================

class ToolExecutor:
    """Executes autopilot tools by calling existing internal functions."""

    def __init__(self, dataset_catalog, dataset_manager,
                 models_dir: Path, books_dir: Path, checkpoints_dir: Path,
                 training_status: dict, active_models: dict,
                 start_training_fn: Callable, stop_training_fn: Callable = None,
                 history=None):
        self.catalog = dataset_catalog
        self.dm = dataset_manager
        self.models_dir = models_dir
        self.books_dir = books_dir
        self.checkpoints_dir = checkpoints_dir
        self.training_status = training_status
        self.active_models = active_models
        self.start_training_fn = start_training_fn
        self.stop_training_fn = stop_training_fn
        # Persistent history store — lazily create default if caller didn't pass one
        if history is None:
            try:
                from model_history import ModelHistory
                history = ModelHistory(self.models_dir.parent / "memory" / "model_history.json")
            except Exception:
                history = None
        self.history = history
        # Track the latest run_id per model so benchmarks attach to the right entry
        self._last_run_id = {}
        # User-defined quality targets (set by LLMAutopilot after start)
        self.quality_spec = {}

    def execute(self, tool_name: str, arguments: dict) -> dict:
        dispatch = {
            "list_catalog": self._list_catalog,
            "search_huggingface": self._search_huggingface,
            "download_dataset": self._download_dataset,
            "create_model": self._create_model,
            "attach_dataset": self._attach_dataset,
            "start_training": self._start_training,
            "stop_training": self._stop_training,
            "check_training_status": self._check_training_status,
            "generate_sample": self._generate_sample,
            "evaluate_model": self._evaluate_model,
            "design_benchmark": self._design_benchmark,
            "run_benchmark": self._run_benchmark,
            "get_model_history": self._get_model_history,
            "compare_runs": self._compare_runs,
            "inspect_dataset": self._inspect_dataset,
            "check_quality_targets": self._check_quality_targets,
            "report_to_user": self._report_to_user,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(**arguments)
        except Exception as e:
            return {"error": f"{tool_name} failed: {str(e)}"}

    def _list_catalog(self, category: str = None, language: str = None) -> dict:
        datasets = self.catalog.get_catalog(category=category, language=language)
        # Trim to essential fields, max 30 results
        items = []
        for d in datasets[:30]:
            items.append({
                "id": d.get("id"),
                "name": d.get("name") or d.get("name_ru"),
                "category": d.get("category"),
                "language": d.get("language"),
                "description": (d.get("description") or d.get("description_en", ""))[:150],
                "size": d.get("size_estimate", "?"),
                "downloaded": d.get("downloaded", False)
            })
        return {"datasets": items, "total": len(datasets)}

    def _search_huggingface(self, query: str, limit: int = 10) -> dict:
        try:
            results = self.catalog.search_huggingface(query, limit=limit)
            return {"results": results[:limit]}
        except Exception as e:
            return {"error": f"HuggingFace search failed: {str(e)}"}

    def _download_dataset(self, dataset_id: str) -> dict:
        try:
            file_path = self.catalog.download_dataset(dataset_id)
            if file_path and file_path.exists():
                filename = file_path.name
                size = file_path.stat().st_size
                # Register in dataset manager
                self.dm.register_dataset(filename, str(file_path), {
                    "source": "catalog",
                    "catalog_id": dataset_id,
                    "size": size
                })
                return {"status": "success", "filename": filename, "size_kb": round(size / 1024, 1)}
            else:
                return {"error": f"Download failed for {dataset_id}"}
        except Exception as e:
            return {"error": f"Download failed: {str(e)}"}

    def _create_model(self, name: str, vocab_size: int = 15000, d_model: int = 256,
                      num_layers: int = 6, num_heads: int = 8, d_ff: int = 1024,
                      max_seq_len: int = 256) -> dict:
        try:
            # Import here to avoid circular
            from model import CustomTransformerLM, count_parameters
            from tokenizer import SimpleTokenizer

            tokenizer = SimpleTokenizer(vocab_size=vocab_size)
            model = CustomTransformerLM(
                vocab_size=vocab_size, d_model=d_model,
                num_layers=num_layers, num_heads=num_heads,
                d_ff=d_ff, max_seq_len=max_seq_len
            )
            params = count_parameters(model)

            model_dir = self.models_dir / name
            model_dir.mkdir(exist_ok=True)

            import torch
            torch.save(model.state_dict(), model_dir / "model.pt")
            tokenizer.save(model_dir / "tokenizer.pkl")

            config = {
                "name": name, "vocab_size": vocab_size, "d_model": d_model,
                "num_layers": num_layers, "num_heads": num_heads,
                "d_ff": d_ff, "max_seq_len": max_seq_len
            }
            with open(model_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            self.active_models[name] = {
                "model": model, "tokenizer": tokenizer, "config": config
            }

            # Record in persistent history
            if self.history is not None:
                try:
                    run_id = self.history.record_run(
                        model_name=name, run_type="create",
                        config=config, notes="Initial model creation (untrained)"
                    )
                    self._last_run_id[name] = run_id
                except Exception:
                    pass

            return {"status": "success", "model_name": name, "parameters": params, "config": config}
        except Exception as e:
            return {"error": f"Failed to create model: {str(e)}"}

    def _attach_dataset(self, model_name: str, dataset_name: str) -> dict:
        try:
            result = self.dm.attach_dataset(model_name, dataset_name)
            if result:
                return {"status": "success", "model_name": model_name, "dataset_name": dataset_name}
            else:
                return {"error": f"Could not attach '{dataset_name}' to '{model_name}'. Check names."}
        except Exception as e:
            return {"error": f"Attach failed: {str(e)}"}

    def _start_training(self, model_name: str, max_iterations: int = 5000,
                        batch_size: int = 16, learning_rate: float = 3e-4) -> dict:
        if self.training_status.get("is_training"):
            return {"error": "Training is already in progress. Wait for it to finish."}

        try:
            # Build a config-like object for train_model_background
            class _Config:
                pass
            config = _Config()
            config.model_name = model_name
            config.max_iterations = max_iterations
            config.batch_size = batch_size
            config.learning_rate = learning_rate
            config.save_every = max(500, max_iterations // 5)
            config.resume_from = None
            config.resume = False
            config.device = "auto"

            # Start in background thread (same as /train endpoint)
            thread = threading.Thread(target=self.start_training_fn, args=(config,), daemon=True)
            thread.start()

            # Record training run in history (finalized when training ends)
            if self.history is not None:
                try:
                    # Collect attached datasets if dataset manager exposes them
                    datasets = []
                    try:
                        attached = self.dm.get_attached(model_name) if hasattr(self.dm, "get_attached") else []
                        datasets = [d.get("filename") if isinstance(d, dict) else str(d) for d in attached]
                    except Exception:
                        pass
                    run_id = self.history.record_run(
                        model_name=model_name, run_type="train",
                        config={"max_iterations": max_iterations,
                                "batch_size": batch_size,
                                "learning_rate": learning_rate},
                        datasets=datasets,
                        notes="Training started"
                    )
                    self._last_run_id[model_name] = run_id
                except Exception:
                    pass

            return {"status": "training_started", "model_name": model_name,
                    "max_iterations": max_iterations, "batch_size": batch_size,
                    "learning_rate": learning_rate}
        except Exception as e:
            return {"error": f"Failed to start training: {str(e)}"}

    def _stop_training(self) -> dict:
        """Stop training in progress."""
        try:
            if self.stop_training_fn:
                self.stop_training_fn()
                return {"status": "stopping", "message": "Training will stop after current batch"}
            return {"error": "No stop function available"}
        except Exception as e:
            return {"error": f"Failed to stop training: {str(e)}"}

    def _check_training_status(self) -> dict:
        s = self.training_status
        return {
            "is_training": s.get("is_training", False),
            "current_iteration": s.get("current_iteration", 0),
            "max_iterations": s.get("max_iterations", 0),
            "current_loss": round(s.get("current_loss", 0), 4),
            "current_reward": round(s.get("current_reward", 0), 4),
            "perplexity": round(s.get("perplexity", 0), 2),
            "tokens_per_sec": round(s.get("tokens_per_sec", 0), 1),
            "eta_seconds": s.get("eta_seconds", -1),
            "model_name": s.get("model_name")
        }

    def _generate_sample(self, model_name: str, prompt: str,
                         max_length: int = 100, temperature: float = 0.8) -> dict:
        try:
            import torch
            import torch.nn.functional as F
            from model import CustomTransformerLM
            from tokenizer import SimpleTokenizer

            model_dir = self.models_dir / model_name
            if not model_dir.exists():
                return {"error": f"Model '{model_name}' not found"}

            # Load model
            if model_name in self.active_models:
                model = self.active_models[model_name]["model"]
                tokenizer = self.active_models[model_name]["tokenizer"]
            else:
                with open(model_dir / "config.json", encoding="utf-8") as f:
                    cfg = json.load(f)
                model = CustomTransformerLM(
                    vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                    num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
                    d_ff=cfg["d_ff"], max_seq_len=cfg["max_seq_len"]
                )
                trained = model_dir / "model_trained.pt"
                weight_file = trained if trained.exists() else model_dir / "model.pt"
                checkpoint = torch.load(weight_file, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()

            tokens = tokenizer.encode(prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                for _ in range(max_length):
                    idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / temperature
                    # Suppress UNK (token 1)
                    logits[:, 1] = -float("Inf")
                    if hasattr(tokenizer, 'token_to_id'):
                        pad_id = tokenizer.token_to_id.get("<PAD>", 0)
                        logits[:, pad_id] = -float("Inf")

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, idx_next], dim=1)

            generated_tokens = idx[0].tolist()
            text = tokenizer.decode(generated_tokens)
            return {"text": text, "tokens_generated": len(generated_tokens) - len(tokens)}
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

    def _evaluate_model(self, model_name: str, language: str = "en", prompts: list = None) -> dict:
        """Run comprehensive quality evaluation on a trained model."""
        try:
            if not prompts or len(prompts) == 0:
                if language == "ru":
                    prompts = ["В этот день", "Она посмотрела", "Когда наступила", "Он шёл по", "Жизнь в городе"]
                else:
                    prompts = ["The morning was", "She walked into", "Once upon a", "He thought about", "In the city"]

            samples = []
            total_unk = 0
            total_tokens = 0
            total_unique_words = set()
            real_word_count = 0
            total_word_count = 0

            for prompt in prompts[:5]:
                result = self._generate_sample(model_name=model_name, prompt=prompt, max_length=80, temperature=0.8)
                if "error" in result:
                    samples.append({"prompt": prompt, "error": result["error"]})
                    continue

                text = result.get("text", "")
                words = text.split()
                unk_count = text.count("<UNK>")
                total_unk += unk_count
                total_tokens += len(words)

                # Check for real words (not <UNK>, <PAD>, not single chars)
                real_words = [w for w in words if not w.startswith("<") and len(w) > 1]
                real_word_count += len(real_words)
                total_word_count += len(words)
                total_unique_words.update(real_words)

                # Check for repetition
                word_set = set(real_words)
                repetition_ratio = 1.0 - (len(word_set) / max(len(real_words), 1))

                samples.append({
                    "prompt": prompt,
                    "generated": text[:300],
                    "unk_count": unk_count,
                    "word_count": len(words),
                    "unique_words": len(word_set),
                    "repetition_ratio": round(repetition_ratio, 2)
                })

            # Overall scores
            unk_ratio = total_unk / max(total_tokens, 1)
            real_word_ratio = real_word_count / max(total_word_count, 1)
            vocabulary_diversity = len(total_unique_words)
            avg_repetition = sum(s.get("repetition_ratio", 0) for s in samples if "error" not in s) / max(len([s for s in samples if "error" not in s]), 1)

            # Determine verdict
            if unk_ratio > 0.5:
                verdict = "TERRIBLE"
                advice = "Model produces mostly <UNK>. Likely no data was attached or tokenizer not trained. Need to attach datasets and retrain."
            elif unk_ratio > 0.2:
                verdict = "BAD"
                advice = "Too many unknown tokens. Need more training data or more iterations."
            elif real_word_ratio < 0.5:
                verdict = "BAD"
                advice = "Too few real words. Model needs more training iterations."
            elif avg_repetition > 0.8:
                verdict = "BAD"
                advice = "Model just repeats the same words. Need more diverse training data or lower temperature."
            elif vocabulary_diversity < 20:
                verdict = "MEDIOCRE"
                advice = "Very limited vocabulary in output. Need more data or more training iterations."
            elif vocabulary_diversity < 50:
                verdict = "OK"
                advice = "Acceptable but could improve with more iterations or data."
            else:
                verdict = "GOOD"
                advice = "Model produces diverse, coherent text."

            # Check training loss
            loss = self.training_status.get("current_loss", 999)
            if loss > 5.0:
                verdict = "TERRIBLE"
                advice = "Training loss is very high ({:.2f}). Model didn't learn. Check if datasets were attached.".format(loss)
            elif loss > 3.5 and verdict in ("OK", "GOOD"):
                verdict = "MEDIOCRE"
                advice = "Loss is still high ({:.2f}). More iterations would improve quality.".format(loss)

            # Composite overall_score in [0, 1] — higher is better
            # Reward: low unk, high real-word ratio, good diversity, not too repetitive
            div_score = min(vocabulary_diversity / 100.0, 1.0)
            rep_score = max(0.0, 1.0 - avg_repetition)
            unk_score = max(0.0, 1.0 - unk_ratio * 2)  # heavy penalty for <UNK>
            overall_score = round(
                0.30 * unk_score +
                0.25 * real_word_ratio +
                0.25 * div_score +
                0.20 * rep_score,
                4
            )

            return {
                "verdict": verdict,
                "advice": advice,
                "overall_score": overall_score,
                "scores": {
                    "unk_ratio": round(unk_ratio, 3),
                    "real_word_ratio": round(real_word_ratio, 3),
                    "vocabulary_diversity": vocabulary_diversity,
                    "avg_repetition": round(avg_repetition, 3),
                    "training_loss": round(loss, 4),
                    "unk_score": round(unk_score, 3),
                    "diversity_score": round(div_score, 3),
                    "non_repetition_score": round(rep_score, 3)
                },
                "samples": samples
            }
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

    # ─── New: benchmark / history / inspection tools ─────────────────────

    def _design_benchmark(self, model_name: str, goal: str, prompts: list,
                          language: str = "en", rubric: str = "") -> dict:
        """Save an LLM-designed benchmark definition tied to this model.
        If user provided custom_prompts in quality_spec, those OVERRIDE the LLM's
        suggestions — the user's prompts are the source of truth."""
        if self.history is None:
            return {"error": "History storage not available"}

        user_prompts = self.quality_spec.get("custom_prompts")
        overridden = False
        if user_prompts:
            prompts = [str(p).strip() for p in user_prompts if str(p).strip()][:10]
            overridden = True

        if not prompts or not isinstance(prompts, list) or len(prompts) < 2:
            return {"error": "Need at least 2 prompts for a meaningful benchmark"}
        prompts = [str(p).strip() for p in prompts if str(p).strip()][:10]

        # Enrich rubric with user's must_include / must_avoid
        rubric_parts = [rubric] if rubric else []
        mi = self.quality_spec.get("must_include")
        ma = self.quality_spec.get("must_avoid")
        if mi:
            rubric_parts.append(f"MUST INCLUDE: {mi}")
        if ma:
            rubric_parts.append(f"MUST AVOID: {ma}")
        full_rubric = " | ".join(rubric_parts)

        try:
            bid = self.history.save_benchmark(model_name, goal, prompts, language, full_rubric)
            return {
                "status": "success",
                "benchmark_id": bid,
                "model_name": model_name,
                "prompt_count": len(prompts),
                "prompts_used": prompts,
                "prompts_overridden_by_user": overridden,
                "language": language,
                "rubric_saved": full_rubric,
                "message": (
                    f"Benchmark '{bid}' saved with {len(prompts)} prompts "
                    + ("(user-supplied)." if overridden else "(LLM-designed).")
                    + " Now call run_benchmark(label='baseline') for the pre-training measurement."
                )
            }
        except Exception as e:
            return {"error": f"Failed to save benchmark: {str(e)}"}

    def _check_quality_targets(self, model_name: str) -> dict:
        """Evaluate the latest benchmarked state of a model against user-supplied
        quality targets from quality_spec. Returns per-criterion pass/fail results."""
        if self.history is None:
            return {"error": "History storage not available"}

        spec = self.quality_spec or {}
        history_snap = self.history.get_history(model_name)
        if not history_snap.get("exists"):
            return {"error": f"No history for '{model_name}'. Run a benchmark first."}

        # Find the most recently benchmarked run
        latest = None
        for r in reversed(history_snap.get("runs", [])):
            bench = r.get("benchmark", {})
            if bench.get("overall_score") is not None:
                latest = r
                break
        if latest is None:
            return {"error": "No benchmarked runs yet. Call run_benchmark first."}

        current_score = latest["benchmark"]["overall_score"]
        current_verdict = latest["benchmark"].get("verdict")

        checks = []
        numeric_passed = True

        target = spec.get("target_score")
        if target is not None:
            passed = current_score >= target
            checks.append({
                "criterion": "target_overall_score",
                "target": target,
                "actual": current_score,
                "passed": passed,
                "detail": f"Need ≥ {target}, got {current_score:.3f}"
            })
            if not passed:
                numeric_passed = False

        min_improvement = spec.get("min_improvement_pct")
        if min_improvement is not None:
            cmp_result = self.history.compare_runs(model_name)
            if "error" in cmp_result:
                checks.append({
                    "criterion": "min_improvement_pct",
                    "target": min_improvement,
                    "actual": None,
                    "passed": False,
                    "detail": f"Cannot compare yet: {cmp_result['error']}"
                })
                numeric_passed = False
            else:
                actual_pct = cmp_result.get("improvement_pct", 0)
                passed = actual_pct >= min_improvement
                checks.append({
                    "criterion": "min_improvement_pct",
                    "target": f"{min_improvement}%",
                    "actual": f"{actual_pct}%",
                    "passed": passed,
                    "detail": f"Need ≥ {min_improvement}% vs baseline, got {actual_pct}%"
                })
                if not passed:
                    numeric_passed = False

        # Qualitative checks — surface samples to LLM for judgment
        if spec.get("must_include"):
            checks.append({
                "criterion": "must_include",
                "target": spec["must_include"],
                "passed": None,
                "detail": "Qualitative — YOU must judge from the benchmark samples whether this holds."
            })
        if spec.get("must_avoid"):
            checks.append({
                "criterion": "must_avoid",
                "target": spec["must_avoid"],
                "passed": None,
                "detail": "Qualitative — YOU must judge from the benchmark samples whether model violates this."
            })

        return {
            "model_name": model_name,
            "current_overall_score": current_score,
            "current_verdict": current_verdict,
            "run_id_checked": latest["run_id"],
            "checks": checks,
            "all_numeric_checks_passed": numeric_passed,
            "has_qualitative_checks": any(c["passed"] is None for c in checks),
            "verdict_summary": (
                "ALL numeric targets met — safe to report final results "
                "(if qualitative checks also hold)." if numeric_passed
                else "FAILURE — at least one numeric target not met. Iterate (more training / better data) or report honest failure to user."
            )
        }

    def _run_benchmark(self, model_name: str, label: str = "") -> dict:
        """Run the most recent benchmark for a model and record the result."""
        if self.history is None:
            return {"error": "History storage not available"}
        latest = self.history.latest_benchmark(model_name)
        if not latest:
            return {"error": f"No benchmark defined for '{model_name}'. Call design_benchmark first."}
        benchmark_id, bdef = latest
        prompts = bdef.get("prompts", [])
        language = bdef.get("language", "en")

        # Run the existing evaluation with the benchmark's prompts
        eval_result = self._evaluate_model(model_name=model_name, language=language, prompts=prompts)
        if "error" in eval_result:
            return eval_result

        # Record on current run if one is active; else create standalone benchmark run
        run_id = self._last_run_id.get(model_name)
        run_type = "attached"
        if run_id:
            # Check if this run already has a benchmark attached; if so, create new entry
            history_snapshot = self.history.get_history(model_name)
            existing_runs = history_snapshot.get("runs", [])
            target_run = next((r for r in existing_runs if r["run_id"] == run_id), None)
            if target_run and target_run.get("benchmark"):
                # Current run already has a benchmark — create fresh standalone entry
                run_id = self.history.record_benchmark_only(
                    model_name, benchmark_id, eval_result,
                    notes=f"label={label}" if label else ""
                )
                self._last_run_id[model_name] = run_id
                run_type = "standalone"
            else:
                # Attach to existing run
                self.history.attach_benchmark_result(model_name, run_id, benchmark_id, eval_result)
        else:
            # No active run — create standalone benchmark entry
            run_id = self.history.record_benchmark_only(
                model_name, benchmark_id, eval_result,
                notes=f"label={label}" if label else ""
            )
            self._last_run_id[model_name] = run_id
            run_type = "standalone"

        return {
            "status": "success",
            "benchmark_id": benchmark_id,
            "run_id": run_id,
            "attachment": run_type,
            "label": label,
            "verdict": eval_result.get("verdict"),
            "overall_score": eval_result.get("overall_score"),
            "scores": eval_result.get("scores"),
            "samples_count": len(eval_result.get("samples", [])),
            "samples_preview": [
                {"prompt": s.get("prompt"), "generated": (s.get("generated") or "")[:160]}
                for s in eval_result.get("samples", [])[:3]
            ]
        }

    def _get_model_history(self, model_name: str) -> dict:
        """Return the full run history for a model."""
        if self.history is None:
            return {"error": "History storage not available"}
        h = self.history.get_history(model_name)
        if not h.get("exists"):
            return {"model_name": model_name, "exists": False,
                    "message": "No history for this model yet (never trained via autopilot)."}
        # Trim run data for LLM consumption — drop verbose samples
        trimmed_runs = []
        for r in h["runs"][-10:]:  # last 10 runs only
            bench = r.get("benchmark", {})
            trimmed_runs.append({
                "run_id": r["run_id"],
                "timestamp": r["timestamp"],
                "type": r["type"],
                "datasets": r.get("datasets", []),
                "training": r.get("training", {}),
                "benchmark_overall": bench.get("overall_score"),
                "benchmark_verdict": bench.get("verdict"),
                "notes": r.get("notes", "")
            })
        return {
            "model_name": model_name,
            "exists": True,
            "created_at": h.get("created_at"),
            "run_count": h.get("run_count"),
            "recent_runs": trimmed_runs,
            "benchmark_count": len(h.get("benchmarks", {}))
        }

    def _compare_runs(self, model_name: str, run_a: str = None, run_b: str = None) -> dict:
        """Compute before/after delta between two benchmarked runs."""
        if self.history is None:
            return {"error": "History storage not available"}
        return self.history.compare_runs(model_name, run_a, run_b)

    def _inspect_dataset(self, filename: str) -> dict:
        """Peek into a downloaded dataset file."""
        try:
            path = self.books_dir / filename
            if not path.exists():
                return {"error": f"File not found: {filename}"}
            size_bytes = path.stat().st_size
            # Read first ~8 KB for preview
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(8000)

            # Stats
            preview = head[:500]
            # Count lines + approx word count across whole file (streaming)
            line_count = 0
            word_count = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_count += 1
                    word_count += len(line.split())

            # Cheap language detection: ratio of Cyrillic to Latin chars
            cyrillic = sum(1 for c in head if "\u0400" <= c <= "\u04FF")
            latin = sum(1 for c in head if ("a" <= c.lower() <= "z"))
            if cyrillic + latin == 0:
                lang_guess = "unknown"
            elif cyrillic > latin * 2:
                lang_guess = "ru"
            elif latin > cyrillic * 2:
                lang_guess = "en"
            else:
                lang_guess = "mixed"

            return {
                "filename": filename,
                "size_kb": round(size_bytes / 1024, 1),
                "line_count": line_count,
                "word_count": word_count,
                "language_guess": lang_guess,
                "preview": preview
            }
        except Exception as e:
            return {"error": f"Inspect failed: {str(e)}"}

    def _report_to_user(self, message: str, is_final: bool = False) -> dict:
        return {"status": "reported", "message": message, "is_final": is_final}


# ============================
# LLM AUTOPILOT (AGENT LOOP)
# ============================

class LLMAutopilot:
    """Main autopilot agent that runs in a background thread."""

    def __init__(self, provider: LLMProvider, tool_executor: ToolExecutor):
        self.provider = provider
        self.provider._log_fn = self._log  # Connect provider logging to autopilot log
        self.executor = tool_executor
        self.state = "idle"
        self.log = []
        self.messages = []
        self.stop_requested = False
        self._thread = None

    def start(self, user_goal: str, time_budget_minutes: int = 0, quality_spec: dict = None):
        """Start the autopilot in a background thread."""
        self.state = "planning"
        self.stop_requested = False
        self.log = []
        self.messages = []
        self.time_budget = time_budget_minutes
        self.quality_spec = quality_spec or {}
        # Share spec with executor so design_benchmark / check_quality_targets can read it
        self.executor.quality_spec = self.quality_spec
        self._thread = threading.Thread(target=self._run, args=(user_goal,), daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_requested = True

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "log": self.log,
            "log_count": len(self.log)
        }

    def _log(self, log_type: str, content: str):
        self.log.append({
            "timestamp": datetime.now().isoformat(),
            "type": log_type,
            "content": content[:2000]  # limit log entry size
        })

    def _run(self, user_goal: str):
        self._log("system", f"Autopilot started. Goal: {user_goal}")

        # Build system prompt with time budget
        if self.time_budget and self.time_budget > 0:
            time_info = f"User set a time limit of {self.time_budget} minutes. Adjust iterations to fit: ~100 iterations/minute on CPU, ~500 iterations/minute on GPU. Leave 2 minutes for evaluation. If time is very short (<10 min), use fewer iterations and smaller model."
        else:
            time_info = "No time limit set. Focus on quality — use enough iterations for good results (at least 3000-5000)."

        system_prompt = SYSTEM_PROMPT.replace("{time_budget_info}", time_info)

        # Build quality spec block for the first user message
        spec = self.quality_spec or {}
        spec_lines = []
        if spec.get("target_score") is not None:
            spec_lines.append(f"- TARGET overall_score: ≥ {spec['target_score']} (composite score in [0..1]; higher=better)")
        if spec.get("min_improvement_pct") is not None:
            spec_lines.append(f"- MINIMUM improvement vs baseline: ≥ {spec['min_improvement_pct']}%")
        if spec.get("must_include"):
            spec_lines.append(f"- Output MUST include/exhibit: {spec['must_include']}")
        if spec.get("must_avoid"):
            spec_lines.append(f"- Output MUST NOT produce: {spec['must_avoid']}")
        if spec.get("custom_prompts"):
            cp = spec["custom_prompts"]
            spec_lines.append(f"- USER-SUPPLIED benchmark prompts ({len(cp)}): {cp}")
            spec_lines.append("  IMPORTANT: when you call design_benchmark, these prompts will be used AUTOMATICALLY — do not invent your own.")

        user_message_parts = [f"GOAL: {user_goal}"]
        if spec_lines:
            user_message_parts.append("\nQUALITY TARGETS (hard requirements from the user):")
            user_message_parts.extend(spec_lines)
            user_message_parts.append(
                "\nYou MUST call check_quality_targets AFTER the post-training benchmark and BEFORE "
                "reporting final results. If any numeric target fails, iterate or honestly report failure. "
                "Do NOT mark is_final=true while numeric targets remain unmet."
            )
        full_user_msg = "\n".join(user_message_parts)
        self._log("system", f"Quality targets: {spec if spec else 'none specified'}")

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_msg}
        ]

        max_turns = 50
        consecutive_errors = 0
        max_consecutive_errors = 3
        empty_turns = 0
        try:
            for turn in range(max_turns):
                if self.stop_requested:
                    self._log("system", "Stopped by user.")
                    self.state = "stopped"
                    return

                # Call LLM
                self._log("thinking", "LLM is analyzing...")
                self.state = "planning" if turn == 0 else "executing"

                try:
                    response = self.provider.chat(self.messages, tools=True)
                    consecutive_errors = 0  # reset on success
                except urllib.error.HTTPError as e:
                    error_body = ""
                    try:
                        error_body = e.read().decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
                    consecutive_errors += 1
                    self._log("error", f"LLM API error {e.code}: {error_body}")
                    if consecutive_errors >= max_consecutive_errors:
                        self._log("system", f"Aborting after {consecutive_errors} consecutive LLM errors.")
                        self.state = "error"
                        return
                    self._log("status", f"Recovering from error ({consecutive_errors}/{max_consecutive_errors}), retrying turn in 5s...")
                    time.sleep(5)
                    continue
                except Exception as e:
                    consecutive_errors += 1
                    self._log("error", f"LLM API error: {str(e)}")
                    if consecutive_errors >= max_consecutive_errors:
                        self._log("error", f"Final traceback: {traceback.format_exc()}")
                        self._log("system", f"Aborting after {consecutive_errors} consecutive LLM errors.")
                        self.state = "error"
                        return
                    self._log("status", f"Recovering from error ({consecutive_errors}/{max_consecutive_errors}), retrying turn in 5s...")
                    time.sleep(5)
                    continue

                # Log text response
                if response.get("content"):
                    self._log("assistant", response["content"])

                # Add assistant message to history
                assistant_msg = self.provider.build_assistant_message(response)
                self.messages.append(assistant_msg)

                # Process tool calls
                if response.get("tool_calls"):
                    for tc in response["tool_calls"]:
                        if self.stop_requested:
                            self._log("system", "Stopped by user.")
                            self.state = "stopped"
                            return

                        tool_name = tc["name"]
                        tool_args = tc["arguments"]
                        tool_id = tc.get("id", tool_name)

                        # Log the call
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in tool_args.items())
                        self._log("tool_call", f"{tool_name}({args_str})")

                        # Execute
                        result = self.executor.execute(tool_name, tool_args)
                        result_str = json.dumps(result, ensure_ascii=False)
                        self._log("tool_result", result_str[:1000])

                        # Add tool result to conversation
                        tool_msg = self.provider.build_tool_result_message(tool_id, result_str, tool_name)
                        self.messages.append(tool_msg)

                        # Handle special cases
                        if tool_name == "report_to_user" and tool_args.get("is_final"):
                            self._log("system", "Autopilot completed successfully!")
                            self.state = "completed"
                            return

                        if tool_name == "start_training" and "error" not in result:
                            self.state = "monitoring"
                            self._monitor_training()
                            # After training ends, continue the loop so LLM can evaluate
                else:
                    # No tool calls. Did the LLM signal a real end?
                    content_text = (response.get("content") or "").strip()
                    stop_reason = response.get("stop_reason")
                    # Real end only if LLM produced final text AND said stop/end_turn
                    if stop_reason in ("stop", "end_turn") and content_text:
                        # If the text looks like a final report — accept it
                        self._log("system", "Autopilot finished (LLM signaled end without report_to_user).")
                        self.state = "completed"
                        return
                    # Otherwise: LLM stalled with empty / non-final response — nudge it
                    empty_turns += 1
                    if empty_turns >= 3:
                        self._log("system", "LLM stalled for 3 turns without tool calls. Stopping.")
                        self.state = "error"
                        return
                    self._log("status", f"LLM produced no tool call (empty turn {empty_turns}/3). Nudging...")
                    self.messages.append({
                        "role": "user",
                        "content": "You did not call any tool. Continue with the next step from your process: call the appropriate tool now (e.g. list_catalog, download_dataset, create_model, start_training, generate_sample, evaluate_model, or report_to_user with is_final=true if you are done)."
                    })
                    continue
                empty_turns = 0  # reset after a turn that had tool calls

            self._log("system", "Reached maximum turns (50). Stopping.")
            self.state = "completed"

        except Exception as e:
            self._log("error", f"Autopilot error: {traceback.format_exc()}")
            self.state = "error"

    def _monitor_training(self):
        """Poll training status with frequent heartbeat + absolute wall-clock cap."""
        self._log("status", "Monitoring training progress...")
        check_interval = 15  # poll every 15s (was 30 — user thought it was frozen)
        last_heartbeat = time.time()
        last_progress_iter = 0
        last_progress_time = time.time()
        last_logged_pct = -1
        started_at = time.time()

        # Wall-clock ceiling: take user's time_budget if set, else 2h
        budget_min = getattr(self, "time_budget", 0) or 0
        wall_clock_limit = (budget_min * 60 + 300) if budget_min > 0 else 7200  # +5min grace

        # Stall detection: no iteration change for N seconds → force-stop
        stall_timeout = 180  # 3 minutes of flatline

        while not self.stop_requested:
            # Sleep in 1-second chunks so stop is responsive
            for _ in range(check_interval):
                if self.stop_requested:
                    self._log("system", "Monitoring stopped by user.")
                    return
                time.sleep(1)

            now = time.time()
            status = self.executor.execute("check_training_status", {})

            if not status.get("is_training"):
                self._log("status", "Training completed!")
                # Finalize training record in history
                self._finalize_training_in_history(status, int(now - started_at))
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"Training has completed. Final status: "
                        f"loss={status.get('current_loss')}, "
                        f"reward={status.get('current_reward')}, "
                        f"iterations={status.get('current_iteration')}, "
                        f"duration={int(now - started_at)}s. "
                        f"Now run the benchmark AGAIN with run_benchmark(label='after_train') "
                        f"and then call compare_runs to prove whether quality improved vs the baseline. "
                        f"Do NOT skip this — the user requires numerical proof of improvement."
                    )
                })
                return

            cur_iter = status.get("current_iteration", 0)
            max_iter = status.get("max_iterations", 0)
            loss = status.get("current_loss", 0)
            reward = status.get("current_reward", 0)

            elapsed = now - started_at

            # Wall-clock watchdog — absolute cap on training time
            if elapsed > wall_clock_limit:
                self._log("status", f"Training exceeded wall-clock limit ({int(elapsed/60)} min). Force-stopping.")
                try:
                    self.executor.execute("stop_training", {})
                except Exception:
                    pass
                time.sleep(3)
                self._finalize_training_in_history(status, int(elapsed))
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"Training was force-stopped after {int(elapsed/60)} minutes (wall-clock limit). "
                        f"Got to iteration {cur_iter}/{max_iter}, loss={loss}, reward={reward}. "
                        f"Run the benchmark with run_benchmark(label='after_force_stop') and then "
                        f"call compare_runs to check if even partial training improved quality vs baseline. "
                        f"Report findings to the user."
                    )
                })
                return

            # Progress / stall tracking
            if cur_iter > last_progress_iter:
                # Real progress — update trackers
                iter_delta = cur_iter - last_progress_iter
                time_delta = now - last_progress_time
                iter_per_sec = iter_delta / max(time_delta, 0.1)
                last_progress_iter = cur_iter
                last_progress_time = now
            else:
                # No progress since last check
                iter_per_sec = 0
                stall_duration = now - last_progress_time
                if stall_duration > stall_timeout:
                    self._log("status", f"No progress for {int(stall_duration)}s at iteration {cur_iter}/{max_iter}. Force-stopping and moving to evaluation.")
                    try:
                        self.executor.execute("stop_training", {})
                    except Exception:
                        pass
                    time.sleep(3)
                    self._finalize_training_in_history(status, int(now - started_at))
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"Training stalled at iteration {cur_iter}/{max_iter} for {int(stall_duration)}s and was stopped. "
                            f"Final loss={loss}, reward={reward}. "
                            f"Run the benchmark with run_benchmark(label='after_stall') and call compare_runs "
                            f"to check if the partial training still improved quality vs baseline. Report findings."
                        )
                    })
                    return

            pct = (cur_iter / max_iter * 100) if max_iter > 0 else 0

            # Log on ANY of these conditions:
            #   - crossed a 2% progress boundary
            #   - 60s elapsed since last heartbeat
            pct_bucket = int(pct // 2) * 2
            should_log = (pct_bucket > last_logged_pct) or (now - last_heartbeat >= 60)

            if should_log:
                # ETA calculation
                if iter_per_sec > 0 and max_iter > cur_iter:
                    eta_sec = (max_iter - cur_iter) / iter_per_sec
                    eta_str = f" | ETA: {int(eta_sec/60)}m{int(eta_sec%60)}s"
                else:
                    eta_str = ""
                speed_str = f" | {iter_per_sec:.1f} it/s" if iter_per_sec > 0 else " | (no progress)"
                self._log("status",
                          f"Training: {cur_iter}/{max_iter} ({pct:.1f}%) | Loss: {loss:.4f} | Reward: {reward:.4f}{speed_str}{eta_str}")
                last_heartbeat = now
                if pct_bucket > last_logged_pct:
                    last_logged_pct = pct_bucket

        self._log("system", "Monitoring stopped by user.")

    def _finalize_training_in_history(self, status: dict, duration_sec: int):
        """Update the training run record with final metrics when training ends."""
        history = getattr(self.executor, "history", None)
        if history is None:
            return
        model_name = status.get("model_name")
        if not model_name:
            return
        last_run_id = self.executor._last_run_id.get(model_name)
        if not last_run_id:
            return
        try:
            with history._lock:
                m = history._data.get("models", {}).get(model_name)
                if not m:
                    return
                for r in m.get("runs", []):
                    if r["run_id"] == last_run_id and r["type"] == "train":
                        r["training"] = {
                            "iterations": status.get("current_iteration", 0),
                            "max_iterations": status.get("max_iterations", 0),
                            "final_loss": round(status.get("current_loss", 0), 4),
                            "final_reward": round(status.get("current_reward", 0), 4),
                            "duration_sec": duration_sec
                        }
                        break
                history._save()
        except Exception:
            pass
