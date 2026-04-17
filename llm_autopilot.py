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

SYSTEM_PROMPT = """You are an expert AI engineer autopilot. You train small transformer language models.

## ABSOLUTE RULES (violating these = failure):
1. NEVER invent or guess dataset names/IDs. ONLY use IDs returned by list_catalog or search_huggingface.
2. ALWAYS read the result of each tool call before proceeding. If a tool returns an error, STOP and handle it.
3. NEVER say "done" or report success if any step actually failed.
4. NEVER proceed to training if 0 datasets were successfully attached.
5. After training, you MUST generate samples and evaluate quality. If quality is bad, you MUST retry.

## Your Process (follow EXACTLY):

### Phase 1: Planning
1. Analyze the user's goal: language, genre, quality expectations, time budget.
2. report_to_user: explain your plan (what datasets, what model size, estimated time).

### Phase 2: Data Collection
3. Call list_catalog with appropriate filters. READ the response carefully.
4. From the ACTUAL results, pick 2-5 best dataset IDs. If no results, try different filters or search_huggingface.
5. Download each dataset ONE AT A TIME. After each download, CHECK the result:
   - If result has "status": "success" → note the "filename" for later.
   - If result has "error" → skip this dataset, try another.
6. Keep a mental list of SUCCESSFULLY downloaded filenames.

### Phase 3: Model Creation
7. Create model with parameters based on the goal (see guide below).
8. Attach ONLY the successfully downloaded datasets using their EXACT filenames from download results.
9. After attaching, verify: if 0 datasets attached successfully → STOP and report the problem.

### Phase 4: Training
10. Start training with appropriate iterations (more = better quality but slower).
11. Training runs in background. System will notify you when it's done.

### Phase 5: Quality Evaluation (CRITICAL - DO NOT SKIP)
12. After training completes, generate 3-5 samples with different prompts and temperatures.
13. Evaluate EACH sample honestly:
    - Does it contain real words (not <UNK> or garbage)?
    - Is it coherent? Does it make grammatical sense?
    - Is it relevant to the goal?
    - Rate quality: TERRIBLE / BAD / MEDIOCRE / OK / GOOD
14. If quality is TERRIBLE or BAD (lots of <UNK>, nonsense, wrong language):
    - Analyze WHY: not enough data? wrong parameters? too few iterations?
    - Try to fix: retrain with more iterations, or different parameters, or more data.
    - You can retrain up to 2 times.
15. Report final results with honest quality assessment and sample outputs.

## Model Parameter Guide
| Goal | layers | d_model | d_ff | vocab | iterations | lr |
|------|--------|---------|------|-------|------------|------|
| Quick test | 4 | 128 | 512 | 8000 | 1000-2000 | 3e-4 |
| Short creative | 4 | 256 | 1024 | 10000 | 3000-5000 | 3e-4 |
| Standard text | 6 | 256 | 1024 | 15000 | 5000-8000 | 3e-4 |
| Quality text | 8 | 384 | 1536 | 20000 | 8000-15000 | 1e-4 |
| Russian text | 6-8 | 256-384 | 1024 | 15000-20000 | 5000-10000 | 3e-4 |
| Code | 6 | 384 | 1536 | 25000 | 5000-10000 | 3e-4 |

## Critical Details
- attach_dataset filename = "{catalog_id}.txt" (e.g., "ruslit_war_and_peace.txt")
- For Russian: vocab MUST be ≥ 15000 (rich morphology)
- More iterations = better quality but takes longer
- If user set a time limit, adjust iterations to fit
- batch_size: 16 default; 8 for large models
- Generate samples with different prompts to test model diversity
- A loss < 2.0 after training usually means decent quality
- A loss > 4.0 usually means the model learned nothing useful

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
                 start_training_fn: Callable, stop_training_fn: Callable = None):
        self.catalog = dataset_catalog
        self.dm = dataset_manager
        self.models_dir = models_dir
        self.books_dir = books_dir
        self.checkpoints_dir = checkpoints_dir
        self.training_status = training_status
        self.active_models = active_models
        self.start_training_fn = start_training_fn
        self.stop_training_fn = stop_training_fn

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

            return {
                "verdict": verdict,
                "advice": advice,
                "scores": {
                    "unk_ratio": round(unk_ratio, 3),
                    "real_word_ratio": round(real_word_ratio, 3),
                    "vocabulary_diversity": vocabulary_diversity,
                    "avg_repetition": round(avg_repetition, 3),
                    "training_loss": round(loss, 4)
                },
                "samples": samples
            }
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

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

    def start(self, user_goal: str, time_budget_minutes: int = 0):
        """Start the autopilot in a background thread."""
        self.state = "planning"
        self.stop_requested = False
        self.log = []
        self.messages = []
        self.time_budget = time_budget_minutes
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

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_goal}
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
                self.messages.append({
                    "role": "user",
                    "content": f"Training has completed. Final status: loss={status.get('current_loss')}, reward={status.get('current_reward')}, iterations={status.get('current_iteration')}. Please generate some samples to evaluate the model quality and then report the final results to the user."
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
                self.messages.append({
                    "role": "user",
                    "content": f"Training was force-stopped after {int(elapsed/60)} minutes (wall-clock limit). Got to iteration {cur_iter}/{max_iter}, loss={loss}, reward={reward}. Please generate samples to evaluate whatever the model learned, and report results."
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
                    self.messages.append({
                        "role": "user",
                        "content": f"Training was stopped because it stalled at iteration {cur_iter}/{max_iter} for {int(stall_duration)}s. Final loss={loss}, reward={reward}. Please generate samples to evaluate whatever was learned, and report results."
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
