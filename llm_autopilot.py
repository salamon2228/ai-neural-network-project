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

SYSTEM_PROMPT = """You are an AI autopilot that automatically trains small transformer language models.
Given a user's goal in natural language, you select the best datasets, create a model with optimal parameters, and train it.

## Your Process
1. ANALYZE the user's goal: what kind of text should the model generate? What language?
2. Use report_to_user to explain your plan before acting.
3. BROWSE the catalog using list_catalog with appropriate category and language filters.
4. SELECT 2-5 datasets that are most relevant. Consider:
   - Language match is CRITICAL (Russian goal → Russian datasets, English → English)
   - Genre/topic match (detective stories need detective datasets, not poetry)
   - Mix of dataset sizes for diversity
5. DOWNLOAD each selected dataset (one at a time, wait for each to complete).
6. CREATE a model with parameters tuned to the goal.
7. ATTACH all downloaded datasets to the model.
8. START training with appropriate parameters.
9. Training runs in the background. Use check_training_status periodically to monitor.
10. After training completes, GENERATE 2-3 sample texts to evaluate quality.
11. Use report_to_user with is_final=true to deliver the final report.

## Model Parameter Guide
- For short creative text: layers=4, d_model=256, iterations=2000-3000, vocab=10000
- For standard text: layers=6, d_model=256, iterations=3000-5000, vocab=15000
- For complex/long text: layers=8, d_model=384, iterations=5000-15000, vocab=20000
- For code: layers=6, d_model=384, vocab=25000+, iterations=5000+
- For Russian text: vocab ≥ 15000 (rich morphology needs larger vocabulary)
- learning_rate: 3e-4 is good default; use 1e-4 for larger models
- batch_size: 16 default; reduce to 8 for large models

## Critical Rules
- The dataset_name for attach_dataset is the FILENAME: "{catalog_id}.txt" (e.g. "gutenberg_sherlock.txt")
- Download datasets ONE AT A TIME (wait for each to finish)
- Always use report_to_user to explain your reasoning before major actions
- If a download fails, skip it and try an alternative
- Choose a descriptive model name related to the goal (e.g. "detective_en", "russian_poet")
- Keep total dataset count between 2-5 for optimal training
- If catalog doesn't have good matches, try search_huggingface
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

    def chat(self, messages: list, tools: list = None) -> dict:
        """Send chat request. Returns {"content": str|None, "tool_calls": list|None, "stop_reason": str}"""
        if self.provider == "anthropic":
            return self._call_anthropic(messages, tools)
        else:
            return self._call_openai(messages, tools)

    def _call_openai(self, messages: list, tools: list = None) -> dict:
        """Call OpenAI or OpenAI-compatible API."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
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

        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        ctx = ssl.create_default_context()

        import re as _re
        for _attempt in range(5):
            try:
                resp = urllib.request.urlopen(req, timeout=120, context=ctx)
                result = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="ignore")
                # Rate limit — wait and retry
                if e.code == 429:
                    wait_match = _re.search(r'try again in (\d+\.?\d*)', error_body)
                    wait_time = float(wait_match.group(1)) + 1 if wait_match else 15
                    import time
                    time.sleep(wait_time)
                    req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
                    continue
                # Groq/Llama: tool_use_failed with failed_generation
                parsed = self._parse_failed_tool_call(error_body)
                if parsed:
                    return parsed
                raise Exception(f"LLM API error {e.code}: {error_body}")
        else:
            raise Exception("LLM API rate limit: too many retries")

        choice = result["choices"][0]
        msg = choice["message"]

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                args = tc["function"].get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": args
                })

        # Fallback: parse tool calls from content if model uses XML-like format
        if not tool_calls and msg.get("content"):
            parsed = self._parse_failed_tool_call(json.dumps({"error": {"failed_generation": msg["content"]}}))
            if parsed and parsed.get("tool_calls"):
                return parsed

        return {
            "content": msg.get("content"),
            "tool_calls": tool_calls,
            "stop_reason": choice.get("finish_reason", "stop"),
            "_raw_message": msg  # keep for conversation history
        }

    def _parse_failed_tool_call(self, error_body: str) -> dict:
        """Parse Groq/Llama failed_generation format: <function=name>{json}</function>"""
        try:
            err = json.loads(error_body)
            failed = err.get("error", {}).get("failed_generation", "")
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
            return response.get("_raw_message", {"role": "assistant", "content": response.get("content", "")})

    def build_tool_result_message(self, tool_call_id: str, result_str: str) -> dict:
        """Build tool result message for conversation history."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_str
        }


# ============================
# TOOL EXECUTOR
# ============================

class ToolExecutor:
    """Executes autopilot tools by calling existing internal functions."""

    def __init__(self, dataset_catalog, dataset_manager,
                 models_dir: Path, books_dir: Path, checkpoints_dir: Path,
                 training_status: dict, active_models: dict,
                 start_training_fn: Callable):
        self.catalog = dataset_catalog
        self.dm = dataset_manager
        self.models_dir = models_dir
        self.books_dir = books_dir
        self.checkpoints_dir = checkpoints_dir
        self.training_status = training_status
        self.active_models = active_models
        self.start_training_fn = start_training_fn

    def execute(self, tool_name: str, arguments: dict) -> dict:
        dispatch = {
            "list_catalog": self._list_catalog,
            "search_huggingface": self._search_huggingface,
            "download_dataset": self._download_dataset,
            "create_model": self._create_model,
            "attach_dataset": self._attach_dataset,
            "start_training": self._start_training,
            "check_training_status": self._check_training_status,
            "generate_sample": self._generate_sample,
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

    def _report_to_user(self, message: str, is_final: bool = False) -> dict:
        return {"status": "reported", "message": message, "is_final": is_final}


# ============================
# LLM AUTOPILOT (AGENT LOOP)
# ============================

class LLMAutopilot:
    """Main autopilot agent that runs in a background thread."""

    def __init__(self, provider: LLMProvider, tool_executor: ToolExecutor):
        self.provider = provider
        self.executor = tool_executor
        self.state = "idle"
        self.log = []
        self.messages = []
        self.stop_requested = False
        self._thread = None

    def start(self, user_goal: str):
        """Start the autopilot in a background thread."""
        self.state = "planning"
        self.stop_requested = False
        self.log = []
        self.messages = []
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

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_goal}
        ]

        max_turns = 50
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
                except urllib.error.HTTPError as e:
                    error_body = ""
                    try:
                        error_body = e.read().decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
                    self._log("error", f"LLM API error {e.code}: {error_body}")
                    self.state = "error"
                    return
                except Exception as e:
                    self._log("error", f"LLM API error: {str(e)}")
                    self.state = "error"
                    return

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
                        tool_msg = self.provider.build_tool_result_message(tool_id, result_str)
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
                    # No tool calls — LLM is done talking
                    if response.get("stop_reason") in ("stop", "end_turn"):
                        self._log("system", "Autopilot finished.")
                        self.state = "completed"
                        return

            self._log("system", "Reached maximum turns (50). Stopping.")
            self.state = "completed"

        except Exception as e:
            self._log("error", f"Autopilot error: {traceback.format_exc()}")
            self.state = "error"

    def _monitor_training(self):
        """Poll training status every 30 seconds until training completes."""
        self._log("status", "Monitoring training progress...")
        check_interval = 30
        last_log_iter = 0

        while not self.stop_requested:
            time.sleep(check_interval)
            status = self.executor.execute("check_training_status", {})

            if not status.get("is_training"):
                self._log("status", "Training completed!")
                # Add training completion to conversation so LLM knows
                self.messages.append({
                    "role": "user",
                    "content": f"Training has completed. Final status: loss={status.get('current_loss')}, reward={status.get('current_reward')}, iterations={status.get('current_iteration')}. Please generate some samples to evaluate the model quality and then report the final results to the user."
                })
                return

            cur_iter = status.get("current_iteration", 0)
            max_iter = status.get("max_iterations", 0)
            loss = status.get("current_loss", 0)
            reward = status.get("current_reward", 0)

            # Log every ~10% progress or every 5 checks
            if cur_iter - last_log_iter > max(max_iter // 10, 100):
                pct = round(cur_iter / max_iter * 100, 1) if max_iter > 0 else 0
                self._log("status", f"Training: {cur_iter}/{max_iter} ({pct}%) | Loss: {loss:.4f} | Reward: {reward:.4f}")
                last_log_iter = cur_iter

        self._log("system", "Monitoring stopped by user.")
