"""
Обновлённый сервер с поддержкой:
- Множественных датасетов
- Каталог датасетов + HuggingFace поиск
- Детальная аналитика обучения
- Сравнение итераций
- Генерация с метриками качества
- Рекомендации по конфигурации
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn.functional as F
import json
import uvicorn
from datetime import datetime
import threading
import shutil
import subprocess
import sys
import re

from model import CustomTransformerLM, count_parameters
from tokenizer import SimpleTokenizer
from azr_trainer_resume import AZRTrainer
from dataset_manager import DatasetManager
from dataset_catalog import DatasetCatalog
from reward_model import RewardComputer
from training_analytics import TrainingAnalytics
from llm_autopilot import LLMAutopilot, LLMProvider, ToolExecutor

app = FastAPI(title="AZR Model Trainer v2")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
BOOKS_DIR = BASE_DIR / "books"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TEMPLATES_DIR = BASE_DIR / "templates"
REPORTS_DIR = BASE_DIR / "reports"

for dir_path in [MODELS_DIR, BOOKS_DIR, CHECKPOINTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Менеджер датасетов
dataset_manager = DatasetManager("datasets_db.json")

# Каталог датасетов
dataset_catalog = DatasetCatalog(BOOKS_DIR)

training_status = {
    "is_training": False,
    "current_iteration": 0,
    "max_iterations": 0,
    "current_loss": 0.0,
    "current_reward": 0.0,
    "model_name": None,
    "history": [],
    "perplexity": 0.0,
    "tokens_per_sec": 0.0,
    "eta_seconds": -1,
    "reward_components": {},
    "memory_mb": 0.0,
}

download_status = {
    "is_downloading": False,
    "dataset_id": None,
    "progress": 0,
    "message": ""
}

active_models = {}
active_trainer = None
active_analytics = None
active_autopilot = None


# === Pydantic Models ===

class ModelConfig(BaseModel):
    name: str
    vocab_size: int = 8000
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256


class TrainingConfig(BaseModel):
    model_name: str
    max_iterations: int = 1000000
    batch_size: int = 16
    learning_rate: float = 3e-4
    save_every: int = 1000
    resume_from: str = None
    resume: bool = False
    device: str = "auto"


class AttachDatasetConfig(BaseModel):
    model_name: str
    dataset_name: str


class DetachDatasetConfig(BaseModel):
    model_name: str
    dataset_name: str


class GenerateConfig(BaseModel):
    model_name: str
    prompt: str
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 40


class GenerateAtCheckpointConfig(BaseModel):
    model_name: str
    checkpoint_iteration: int
    prompt: str
    max_length: int = 100
    temperature: float = 0.8


class CompareConfig(BaseModel):
    model_name: str
    prompt: str
    iterations: List[int]
    max_length: int = 100


class AutopilotConfig(BaseModel):
    goal: str
    provider: str = "openai"
    api_key: Optional[str] = ""
    endpoint: Optional[str] = None
    model: Optional[str] = None
    time_budget: Optional[int] = 0


# === UI ===

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = TEMPLATES_DIR / "index_complete.html"
    if not html_file.exists():
        html_file = TEMPLATES_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    return "<h1>AZR Model Trainer v2</h1><p>Template not found. Run build_complete_interface.py</p>"


# === Model CRUD ===

@app.post("/create_model")
async def create_model(config: ModelConfig):
    try:
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

        model = CustomTransformerLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len
        )

        params = count_parameters(model)

        model_dir = MODELS_DIR / config.name
        model_dir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), model_dir / "model.pt")
        tokenizer.save(model_dir / "tokenizer.pkl")

        with open(model_dir / "config.json", "w") as f:
            json.dump(config.dict(), f, indent=2)

        active_models[config.name] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": config.dict()
        }

        return {
            "status": "success",
            "message": f"Model '{config.name}' created successfully",
            "parameters": params,
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            config_file = model_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

                model_info = dataset_manager.get_model_info(model_dir.name)

                model_size = 0
                trained_model = model_dir / "model_trained.pt"
                base_model = model_dir / "model.pt"

                if trained_model.exists():
                    model_size = trained_model.stat().st_size
                elif base_model.exists():
                    model_size = base_model.stat().st_size

                model_size_mb = round(model_size / (1024 * 1024), 2)

                models.append({
                    "name": model_dir.name,
                    "config": config,
                    "datasets": model_info["datasets"] if model_info else [],
                    "total_datasets": model_info["total_datasets"] if model_info else 0,
                    "size_mb": model_size_mb,
                    "trained": trained_model.exists()
                })
    return {"models": models}


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Удалить модель и все её файлы"""
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    try:
        shutil.rmtree(model_dir)
        # Удаляем чекпоинты если есть
        checkpoint_dir = CHECKPOINTS_DIR / model_name
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        return {"status": "deleted", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Dataset Upload & Management ===

@app.post("/upload_book")
async def upload_book(file: UploadFile = File(...)):
    try:
        file_path = BOOKS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Определяем формат
        metadata = None
        converted_name = file.filename
        try:
            fmt_info = dataset_catalog.detect_format(file_path)
            metadata = fmt_info

            # Конвертируем если не txt
            if fmt_info["format"] != "txt":
                converted = dataset_catalog.convert_to_txt(file_path, fmt_info["format"])
                if converted != file_path:
                    file_path = converted
                    converted_name = converted.name
                    metadata["converted"] = True
        except Exception as conv_err:
            metadata = metadata or {}
            metadata["conversion_error"] = str(conv_err)

        dataset_manager.register_dataset(converted_name, str(file_path), metadata=metadata)

        return {
            "status": "success",
            "filename": converted_name,
            "original_filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "format": metadata.get("format", "txt") if metadata else "txt"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/books")
async def list_books():
    datasets = dataset_manager.get_available_datasets()
    return {"books": datasets}


@app.post("/attach_dataset")
async def attach_dataset(config: AttachDatasetConfig):
    try:
        success = dataset_manager.attach_dataset(config.model_name, config.dataset_name)
        if success:
            return {
                "status": "success",
                "message": f"Dataset '{config.dataset_name}' attached to '{config.model_name}'"
            }
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detach_dataset")
async def detach_dataset(config: DetachDatasetConfig):
    try:
        success = dataset_manager.detach_dataset(config.model_name, config.dataset_name)
        if success:
            return {
                "status": "success",
                "message": f"Dataset '{config.dataset_name}' detached from '{config.model_name}'"
            }
        else:
            raise HTTPException(status_code=404, detail="Not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_dataset/{dataset_name}")
async def delete_dataset_endpoint(dataset_name: str):
    """Удалить датасет из системы и с диска"""
    try:
        # Получить путь к файлу до удаления из БД
        file_path = None
        if dataset_name in dataset_manager.db["datasets"]:
            file_path = dataset_manager.db["datasets"][dataset_name].get("path")

        success = dataset_manager.delete_dataset(dataset_name)
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Удалить файл с диска
        if file_path:
            p = Path(file_path)
            if p.exists():
                p.unlink()

        return {"status": "success", "message": f"Dataset '{dataset_name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_datasets/{model_name}")
async def get_model_datasets(model_name: str):
    try:
        attached = dataset_manager.get_attached_datasets(model_name)
        attached_info = []
        for ds_name in attached:
            if ds_name in dataset_manager.db["datasets"]:
                info = dataset_manager.db["datasets"][ds_name]
                attached_info.append({
                    "name": ds_name,
                    "size": info["size"],
                    "path": info["path"]
                })

        available = dataset_manager.get_available_datasets(model_name)

        return {
            "attached": attached_info,
            "available": available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Dataset Catalog ===

@app.get("/dataset_catalog")
async def get_dataset_catalog(category: str = None, language: str = None):
    return {"datasets": dataset_catalog.get_catalog(category, language)}


@app.get("/dataset_catalog/categories")
async def get_catalog_categories():
    return {"categories": dataset_catalog.get_categories()}


@app.post("/dataset_catalog/download/{dataset_id}")
async def download_catalog_dataset(dataset_id: str):
    global download_status

    if download_status["is_downloading"]:
        raise HTTPException(status_code=400, detail="Another download in progress")

    info = dataset_catalog.get_dataset_info(dataset_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    def do_download():
        global download_status
        download_status["is_downloading"] = True
        download_status["dataset_id"] = dataset_id

        def progress_cb(status):
            download_status["progress"] = status.get("progress", 0)
            download_status["message"] = status.get("message", "")

        try:
            file_path = dataset_catalog.download_dataset(dataset_id, progress_callback=progress_cb)
            if file_path:
                # Регистрируем в менеджере датасетов
                dataset_manager.register_dataset(
                    file_path.name, str(file_path),
                    metadata={
                        "source": "catalog",
                        "catalog_id": dataset_id,
                        "description": info.get("description", ""),
                        "format": "txt"
                    }
                )
                download_status["message"] = f"Downloaded: {file_path.name}"
        except Exception as e:
            download_status["message"] = f"Error: {e}"
            download_status["progress"] = -1
        finally:
            download_status["is_downloading"] = False

    thread = threading.Thread(target=do_download)
    thread.start()

    return {"status": "downloading", "dataset_id": dataset_id, "name": info["name"]}


@app.get("/dataset_catalog/download_status")
async def get_download_status():
    return download_status


@app.get("/dataset_catalog/preview/{dataset_id}")
async def preview_catalog_dataset(dataset_id: str, lines: int = 20):
    return dataset_catalog.preview_dataset(dataset_id, lines)


@app.get("/dataset_catalog/search_hf")
async def search_huggingface(query: str, limit: int = 10):
    results = dataset_catalog.search_huggingface(query, limit)
    return {"results": results}


@app.post("/dataset_catalog/add_custom")
async def add_custom_dataset(name: str = Form(...), url: str = Form(...), language: str = Form("auto")):
    """Добавить свой датасет по URL"""
    try:
        custom_entry = dataset_catalog.add_custom_url(name, url, language)
        return {"status": "added", "dataset": custom_entry}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Training ===

def train_model_background(config: TrainingConfig):
    global training_status, active_trainer, active_analytics

    try:
        training_status["is_training"] = True
        training_status["model_name"] = config.model_name
        training_status["max_iterations"] = config.max_iterations

        model_dir = MODELS_DIR / config.model_name

        # Загружаем модель
        if config.model_name in active_models:
            model = active_models[config.model_name]["model"]
            tokenizer = active_models[config.model_name]["tokenizer"]
        else:
            with open(model_dir / "config.json") as f:
                model_config = json.load(f)

            model = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )
            model.load_state_dict(torch.load(model_dir / "model.pt"))
            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")

        # Загружаем ВСЕ прикреплённые датасеты
        print(f"\nLoading attached datasets for model '{config.model_name}'...")
        texts = dataset_manager.load_attached_texts(config.model_name)

        if len(texts) == 0:
            print("ERROR: No attached datasets! Attach datasets first.")
            training_status["is_training"] = False
            return

        print(f"Loaded {len(texts)} text chunks from attached datasets")

        # Проверяем нужно ли переобучить токенизатор
        current_datasets = set(dataset_manager.get_attached_datasets(config.model_name))
        tokenizer_datasets = set(tokenizer.get_trained_datasets())
        need_retrain = False
        preserve_mode = False

        is_pretrained = (len(tokenizer.token_to_id) > 1000 and len(tokenizer_datasets) == 0)

        if is_pretrained:
            print(f"Pretrained model detected ({len(tokenizer.token_to_id)} tokens)")
            need_retrain = False
        elif len(tokenizer.token_to_id) <= len(tokenizer.special_tokens):
            need_retrain = True
            preserve_mode = False
            print("Tokenizer is empty, will train...")
        elif current_datasets != tokenizer_datasets:
            need_retrain = True
            preserve_mode = True
            print("Datasets changed, updating tokenizer...")
        else:
            print(f"Tokenizer already trained on current datasets")

        if need_retrain:
            print(f"Training tokenizer on {len(texts)} text chunks...")
            batch_size_tok = 100
            all_texts_combined = []
            for i in range(0, len(texts), batch_size_tok):
                batch = texts[i:i + batch_size_tok]
                all_texts_combined.append(" ".join(batch))

            tokenizer.train(all_texts_combined, preserve_existing=preserve_mode)
            print(f"Tokenizer trained! Vocabulary size: {len(tokenizer.token_to_id)} tokens")
            tokenizer.save(model_dir / "tokenizer.pkl", trained_on_datasets=list(current_datasets))

        # Определяем устройство
        if config.device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif config.device == "cuda":
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, using CPU")
                device = 'cpu'
            else:
                device = 'cuda'
        else:
            device = 'cpu'

        print(f"Using device: {device}")
        model = model.to(device)
        training_status["device"] = device

        # Callback для обновления статуса
        def update_status(status_dict):
            training_status.update(status_dict)

        # Создаём RewardComputer с референсными текстами
        reward_computer = RewardComputer(tokenizer, reference_texts=texts[:100])

        # Создаём аналитику
        reports_dir = REPORTS_DIR / config.model_name
        analytics = TrainingAnalytics(reports_dir)
        active_analytics = analytics

        # Создаём trainer
        active_trainer = AZRTrainer(
            model, tokenizer,
            device=device,
            status_callback=update_status,
            reward_computer=reward_computer,
            analytics=analytics
        )

        # Обучение
        history = active_trainer.train_continuous(
            texts=texts,
            max_iterations=config.max_iterations,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            save_every=config.save_every,
            checkpoint_dir=CHECKPOINTS_DIR / config.model_name,
            resume_from=config.resume_from
        )

        # Сохраняем финальную модель
        torch.save(model.state_dict(), model_dir / "model_trained.pt")

        training_status["is_training"] = False
        training_status["history"] = history

    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


@app.post("/train")
async def start_training(config: TrainingConfig):
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")

    model_dir = MODELS_DIR / config.model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{config.model_name}' not found")

    attached = dataset_manager.get_attached_datasets(config.model_name)
    if len(attached) == 0:
        raise HTTPException(status_code=400, detail="No datasets attached to model. Attach datasets first!")

    # Auto-find latest checkpoint for resume
    if config.resume and not config.resume_from:
        checkpoint_dir = CHECKPOINTS_DIR / config.model_name
        if checkpoint_dir.exists():
            # Find latest checkpoint (paused > regular > error)
            candidates = []
            for pattern in ["model_paused_*.pt", "model_iter_*.pt", "model_interrupted_*.pt", "model_error_*.pt"]:
                candidates.extend(checkpoint_dir.glob(pattern))
            if candidates:
                latest = max(candidates, key=lambda f: f.stat().st_mtime)
                config.resume_from = str(latest)
                print(f"Auto-resuming from: {latest.name}")

    thread = threading.Thread(target=train_model_background, args=(config,))
    thread.start()

    return {
        "status": "success",
        "message": f"Training started with {len(attached)} datasets" + (f" (resuming from checkpoint)" if config.resume_from else ""),
        "datasets": attached,
        "resumed": bool(config.resume_from),
        "config": config.model_dump()
    }


@app.post("/stop_training")
async def stop_training():
    global active_trainer
    if active_trainer:
        active_trainer.stop_training()
        return {"status": "stopping", "message": "Training will pause after current batch"}
    return {"status": "not training"}


@app.get("/training_status")
async def get_training_status():
    return training_status


@app.post("/generate_live")
async def generate_live(config: GenerateConfig):
    """Генерация текста из текущей обучающейся модели (или последней обученной)"""
    global active_trainer
    try:
        model = None
        tokenizer = None

        # Пробуем использовать модель из активного trainer'а
        if active_trainer and active_trainer.model is not None:
            model = active_trainer.model
            tokenizer = active_trainer.tokenizer
        else:
            # Fallback: загрузить из файла
            model_dir = MODELS_DIR / config.model_name
            if not model_dir.exists():
                raise HTTPException(status_code=404, detail="Model not found")
            with open(model_dir / "config.json") as f:
                model_config = json.load(f)
            model = CustomTransformerLM(**{k: model_config[k] for k in ['vocab_size', 'd_model', 'num_layers', 'num_heads', 'd_ff', 'max_seq_len']})
            trained = model_dir / "model_trained.pt"
            weights = trained if trained.exists() else model_dir / "model.pt"
            ckpt = torch.load(weights, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)

        model.eval()
        device = next(model.parameters()).device
        tokens = tokenizer.encode(config.prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=config.max_length,
                                      temperature=config.temperature, top_k=config.top_k or 40)

        gen_text = tokenizer.decode(generated[0].cpu().tolist())

        # Return model to training mode if it was being trained
        if training_status.get("is_training"):
            model.train()

        return {
            "status": "success",
            "generated_text": gen_text,
            "iteration": training_status.get("current_iteration", 0),
            "is_training": training_status.get("is_training", False)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Training Analytics ===

@app.get("/training_analytics")
async def get_training_analytics():
    if active_analytics:
        return active_analytics.get_summary()
    return {"error": "No active analytics session"}


@app.get("/training_analytics/reports")
async def get_analytics_reports():
    if active_analytics:
        return {"reports": active_analytics.get_all_reports()}
    return {"reports": []}


@app.get("/training_analytics/benchmarks")
async def get_benchmark_results():
    if active_analytics:
        return {"benchmarks": active_analytics.get_benchmark_history()}
    return {"benchmarks": []}


@app.get("/training_analytics/compare/{iter_a}/{iter_b}")
async def compare_iterations(iter_a: int, iter_b: int):
    if active_analytics:
        comparison = active_analytics.compare_iterations(iter_a, iter_b)
        if comparison:
            return comparison
        return {"error": f"Iterations {iter_a} or {iter_b} not found in reports"}
    return {"error": "No active analytics session"}


# === Checkpoints & Comparison ===

@app.get("/checkpoints/{model_name}")
async def list_checkpoints(model_name: str):
    checkpoint_dir = CHECKPOINTS_DIR / model_name
    if not checkpoint_dir.exists():
        return {"checkpoints": []}

    checkpoints = []
    for f in sorted(checkpoint_dir.glob("model_iter_*.pt")):
        try:
            # Извлекаем номер итерации из имени файла
            iter_num = int(f.stem.split("_")[-1])
            checkpoints.append({
                "iteration": iter_num,
                "filename": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "timestamp": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
        except (ValueError, IndexError):
            continue

    return {"checkpoints": sorted(checkpoints, key=lambda x: x["iteration"])}


@app.post("/generate_at_checkpoint")
async def generate_at_checkpoint(config: GenerateAtCheckpointConfig):
    try:
        model_dir = MODELS_DIR / config.model_name
        checkpoint_dir = CHECKPOINTS_DIR / config.model_name

        with open(model_dir / "config.json") as f:
            model_config = json.load(f)

        # Ищем чекпоинт
        checkpoint_file = checkpoint_dir / f"model_iter_{config.checkpoint_iteration}.pt"
        if not checkpoint_file.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint at iteration {config.checkpoint_iteration} not found")

        model = CustomTransformerLM(
            vocab_size=model_config["vocab_size"],
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            max_seq_len=model_config["max_seq_len"]
        )

        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        tokens = tokenizer.encode(config.prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=config.max_length,
                                      temperature=config.temperature, top_k=40)

        gen_tokens = generated[0].cpu().tolist()
        gen_text = tokenizer.decode(gen_tokens)

        # Оценка качества
        reward_computer = RewardComputer(tokenizer)
        quality = reward_computer.compute_reward(gen_text)

        return {
            "status": "success",
            "iteration": config.checkpoint_iteration,
            "prompt": config.prompt,
            "generated_text": gen_text,
            "quality_metrics": quality
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare_generations")
async def compare_generations(config: CompareConfig):
    try:
        model_dir = MODELS_DIR / config.model_name
        checkpoint_dir = CHECKPOINTS_DIR / config.model_name

        with open(model_dir / "config.json") as f:
            model_config = json.load(f)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        reward_computer = RewardComputer(tokenizer)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        results = []

        for iteration in config.iterations[:5]:  # Макс 5
            checkpoint_file = checkpoint_dir / f"model_iter_{iteration}.pt"
            if not checkpoint_file.exists():
                results.append({
                    "iteration": iteration,
                    "error": "Checkpoint not found"
                })
                continue

            model = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )

            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(device)
            model.eval()

            tokens = tokenizer.encode(config.prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                generated = model.generate(idx, max_new_tokens=config.max_length,
                                          temperature=0.8, top_k=40)

            gen_tokens = generated[0].cpu().tolist()
            gen_text = tokenizer.decode(gen_tokens)
            quality = reward_computer.compute_reward(gen_text)

            results.append({
                "iteration": iteration,
                "text": gen_text,
                "quality": quality
            })

            del model

        return {"prompt": config.prompt, "comparisons": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Generation ===

@app.post("/generate")
async def generate_text(config: GenerateConfig):
    try:
        model_dir = MODELS_DIR / config.model_name

        if config.model_name in active_models:
            model = active_models[config.model_name]["model"]
            tokenizer = active_models[config.model_name]["tokenizer"]
        else:
            with open(model_dir / "config.json") as f:
                model_config = json.load(f)

            model = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )

            trained_model = model_dir / "model_trained.pt"
            if trained_model.exists():
                checkpoint = torch.load(trained_model, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(torch.load(model_dir / "model.pt", map_location='cpu'))

            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        input_tokens = tokenizer.encode(config.prompt)
        idx = torch.tensor([input_tokens], dtype=torch.long, device=device)

        # Генерация с отслеживанием confidence per token
        token_details = []
        current_idx = idx.clone()

        with torch.no_grad():
            for _ in range(config.max_length):
                idx_cond = current_idx if current_idx.size(1) <= model.max_seq_len else current_idx[:, -model.max_seq_len:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / config.temperature

                if config.top_k:
                    v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                chosen_prob = probs[0, idx_next[0, 0]].item()

                # Топ-3 альтернативы
                top3_probs, top3_ids = torch.topk(probs[0], min(3, probs.size(-1)))
                alternatives = []
                for p, tid in zip(top3_probs.tolist(), top3_ids.tolist()):
                    token_str = tokenizer.id_to_token.get(tid, f"<{tid}>") if hasattr(tokenizer, 'id_to_token') else str(tid)
                    alternatives.append({"token": token_str, "prob": round(p, 4)})

                token_id = idx_next[0, 0].item()
                token_str = tokenizer.id_to_token.get(token_id, f"<{token_id}>") if hasattr(tokenizer, 'id_to_token') else str(token_id)

                token_details.append({
                    "token": token_str,
                    "confidence": round(chosen_prob, 4),
                    "top_alternatives": alternatives
                })

                current_idx = torch.cat((current_idx, idx_next), dim=1)

        generated_tokens = current_idx[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_tokens)

        # Метрики качества
        reward_computer = RewardComputer(tokenizer)
        quality = reward_computer.compute_reward(generated_text)

        return {
            "status": "success",
            "prompt": config.prompt,
            "generated_text": generated_text,
            "token_details": token_details,
            "quality_metrics": quality,
            "tokens_generated": len(token_details),
        }
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_before_after")
async def generate_before_after(config: GenerateConfig):
    """Сравнение генерации до и после обучения"""
    try:
        model_dir = MODELS_DIR / config.model_name

        with open(model_dir / "config.json") as f:
            model_config = json.load(f)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        reward_computer = RewardComputer(tokenizer)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def generate_from_weights(weights_path):
            m = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )
            ckpt = torch.load(weights_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                m.load_state_dict(ckpt['model_state_dict'])
            else:
                m.load_state_dict(ckpt)
            m = m.to(device)
            m.eval()

            tokens = tokenizer.encode(config.prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = m.generate(idx, max_new_tokens=config.max_length,
                                temperature=config.temperature, top_k=config.top_k)
            text = tokenizer.decode(gen[0].cpu().tolist())
            quality = reward_computer.compute_reward(text)
            del m
            return text, quality

        # До обучения
        before_text, before_quality = generate_from_weights(model_dir / "model.pt")

        # После обучения
        trained_path = model_dir / "model_trained.pt"
        if trained_path.exists():
            after_text, after_quality = generate_from_weights(trained_path)
        else:
            after_text = before_text
            after_quality = before_quality

        # Считаем улучшение
        improvement = {
            "reward_delta": round(after_quality["total"] - before_quality["total"], 4),
            "components_delta": {}
        }
        for key in after_quality.get("components", {}):
            before_val = before_quality.get("components", {}).get(key, 0)
            after_val = after_quality["components"][key]
            improvement["components_delta"][key] = round(after_val - before_val, 4)

        return {
            "prompt": config.prompt,
            "before": {"text": before_text, "quality": before_quality},
            "after": {"text": after_text, "quality": after_quality},
            "improvement": improvement
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Download / Upload Model ===

@app.get("/download_model/{model_name}")
async def download_model(model_name: str):
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    trained_model = model_dir / "model_trained.pt"
    if trained_model.exists():
        return FileResponse(trained_model, filename=f"{model_name}_trained.pt",
                          media_type="application/octet-stream")
    else:
        return FileResponse(model_dir / "model.pt", filename=f"{model_name}.pt",
                          media_type="application/octet-stream")


@app.post("/upload_model/{model_name}")
async def upload_model(model_name: str, file: UploadFile = File(...)):
    """Загрузить модель: если модель существует — обновляет веса, если нет — создаёт из чекпоинта"""
    try:
        model_dir = MODELS_DIR / model_name
        is_new = not model_dir.exists()

        if is_new:
            model_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем файл
        model_path = model_dir / "model_trained.pt"
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # Загружаем чекпоинт
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

            if is_new:
                # Новая модель — извлекаем конфиг из чекпоинта или определяем из state_dict
                state_dict = ckpt
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                    model_config = ckpt.get('model_config', {})
                else:
                    model_config = {}

                # Определяем параметры из весов если конфиг не найден
                if not model_config:
                    emb_key = next((k for k in state_dict if 'embedding' in k and 'weight' in k), None)
                    if emb_key:
                        vocab_size, d_model = state_dict[emb_key].shape
                    else:
                        vocab_size, d_model = 10000, 256
                    # Считаем уникальные номера блоков (blocks.0, blocks.1, ...)
                    block_ids = set()
                    for k in state_dict:
                        m = re.match(r'blocks\.(\d+)\.', k)
                        if m:
                            block_ids.add(int(m.group(1)))
                    num_layers = len(block_ids) if block_ids else 4
                    # Определяем max_seq_len из position_embedding
                    pos_key = next((k for k in state_dict if 'position' in k and 'weight' in k), None)
                    max_seq_len = int(state_dict[pos_key].shape[0]) if pos_key else 256
                    # Определяем d_ff из первого FF слоя
                    ff_key = next((k for k in state_dict if '.ff.linear1.weight' in k), None)
                    d_ff = int(state_dict[ff_key].shape[0]) if ff_key else int(d_model) * 4
                    model_config = {
                        'vocab_size': int(vocab_size), 'd_model': int(d_model),
                        'num_layers': max(num_layers, 1), 'num_heads': max(int(d_model) // 64, 1),
                        'd_ff': d_ff, 'max_seq_len': max_seq_len, 'name': model_name
                    }

                # Создаём модель и проверяем что веса подходят
                model = CustomTransformerLM(**{k: model_config[k] for k in ['vocab_size', 'd_model', 'num_layers', 'num_heads', 'd_ff', 'max_seq_len']})
                model.load_state_dict(state_dict)

                # Сохраняем конфиг и initial weights
                with open(model_dir / "config.json", 'w', encoding='utf-8') as f:
                    json.dump(model_config, f, indent=2, ensure_ascii=False)
                torch.save(state_dict, model_dir / "model.pt")

                # Ищем токенизатор: 1) в чекпоинте, 2) от существующей модели, 3) пустой
                tokenizer_found = False
                tokenizer_source = "empty"

                # 1) Токенизатор в чекпоинте (если checkpoint, а не state_dict)
                if isinstance(ckpt, dict) and 'tokenizer_data' in ckpt:
                    tokenizer = SimpleTokenizer(vocab_size=model_config['vocab_size'])
                    tok_data = ckpt['tokenizer_data']
                    tokenizer.token_to_id = tok_data.get('token_to_id', {})
                    tokenizer.id_to_token = {int(k): v for k, v in tok_data.get('id_to_token', {}).items()}
                    tokenizer.save(model_dir / "tokenizer.pkl")
                    tokenizer_found = True
                    tokenizer_source = "checkpoint"

                # 2) Копируем от существующей модели с таким же vocab_size
                if not tokenizer_found:
                    for existing_model in MODELS_DIR.iterdir():
                        if existing_model.is_dir() and existing_model.name != model_name:
                            tok_path = existing_model / "tokenizer.pkl"
                            cfg_path = existing_model / "config.json"
                            if tok_path.exists() and cfg_path.exists():
                                try:
                                    with open(cfg_path, 'r', encoding='utf-8') as f:
                                        ecfg = json.load(f)
                                    if ecfg.get('vocab_size') == model_config['vocab_size']:
                                        existing_tok = SimpleTokenizer.load(tok_path)
                                        if len(existing_tok.token_to_id) > len(SimpleTokenizer(model_config['vocab_size']).token_to_id):
                                            shutil.copy2(tok_path, model_dir / "tokenizer.pkl")
                                            tokenizer_found = True
                                            tokenizer_source = existing_model.name
                                            break
                                except Exception:
                                    pass

                # 3) Пустой токенизатор (последний вариант)
                if not tokenizer_found:
                    tokenizer = SimpleTokenizer(vocab_size=model_config['vocab_size'])
                    tokenizer.save(model_dir / "tokenizer.pkl")
                    tokenizer_source = "empty"

                tok_msg = ""
                if tokenizer_source == "empty":
                    tok_msg = " ⚠️ No tokenizer found — upload tokenizer.pkl separately"
                elif tokenizer_source == "checkpoint":
                    tok_msg = " ✅ Tokenizer from checkpoint"
                else:
                    tok_msg = f" ✅ Tokenizer from '{tokenizer_source}'"

                return {
                    "status": "success",
                    "message": f"Model '{model_name}' created ({model_config['d_model']}d / {model_config['num_layers']} layers / {model_config['vocab_size']} vocab).{tok_msg}",
                    "config": model_config,
                    "tokenizer_source": tokenizer_source,
                    "size": model_path.stat().st_size
                }
            else:
                # Существующая модель — обновляем веса
                with open(model_dir / "config.json") as f:
                    model_config = json.load(f)

                state_dict = ckpt
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']

                model = CustomTransformerLM(**{k: model_config[k] for k in ['vocab_size', 'd_model', 'num_layers', 'num_heads', 'd_ff', 'max_seq_len']})
                model.load_state_dict(state_dict)

                if model_name in active_models:
                    active_models[model_name]["model"] = model

                return {
                    "status": "success",
                    "message": f"Model '{model_name}' weights updated",
                    "size": model_path.stat().st_size
                }
        except Exception as e:
            if is_new and model_dir.exists():
                shutil.rmtree(model_dir)
            elif model_path.exists():
                model_path.unlink()
            raise HTTPException(status_code=400, detail=f"Invalid model file: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_tokenizer/{model_name}")
async def upload_tokenizer(model_name: str, file: UploadFile = File(...)):
    """Загрузить токенизатор для модели"""
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    tok_path = model_dir / "tokenizer.pkl"
    with open(tok_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Проверяем что файл валидный
    try:
        tokenizer = SimpleTokenizer.load(tok_path)
        vocab_count = len(tokenizer.token_to_id)
        return {
            "status": "success",
            "message": f"Tokenizer uploaded for '{model_name}' ({vocab_count} tokens)",
            "vocab_count": vocab_count
        }
    except Exception as e:
        tok_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid tokenizer file: {str(e)}")


# === System Info ===

@app.get("/dataset_stats")
async def get_dataset_stats():
    return dataset_manager.get_stats()


@app.get("/device_info")
async def get_device_info():
    cuda_available = torch.cuda.is_available()

    info = {
        "cpu": {
            "available": True,
            "name": "CPU",
            "cores": torch.get_num_threads()
        },
        "cuda": {
            "available": cuda_available,
            "name": torch.cuda.get_device_name(0) if cuda_available else None,
            "count": torch.cuda.device_count() if cuda_available else 0,
            "memory": None
        },
        "current_device": "cuda" if cuda_available else "cpu",
        "recommendation": "GPU (CUDA)" if cuda_available else "CPU (slow, GPU recommended)"
    }

    if cuda_available:
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["cuda"]["memory"] = f"{total_memory:.1f} GB"
        except Exception:
            pass

    return info


@app.get("/hardware_recommendation")
async def get_hardware_recommendation():
    """Рекомендация конфигурации на основе оборудования"""
    has_gpu = torch.cuda.is_available()
    gpu_mem_gb = 0

    if has_gpu:
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        ram_gb = 8  # Default assumption

    if has_gpu and gpu_mem_gb >= 8:
        preset = "quality"
        gpu_name = torch.cuda.get_device_name(0)
        reason_key = "hw_gpu_large"
        reason_params = {"name": gpu_name, "mem": f"{gpu_mem_gb:.0f}"}
        reason = f"GPU {gpu_name} ({gpu_mem_gb:.0f} GB) — can train large models"
    elif has_gpu and gpu_mem_gb >= 4:
        preset = "standard"
        gpu_name = torch.cuda.get_device_name(0)
        reason_key = "hw_gpu_standard"
        reason_params = {"name": gpu_name, "mem": f"{gpu_mem_gb:.0f}"}
        reason = f"GPU {gpu_name} ({gpu_mem_gb:.0f} GB) — standard models"
    elif has_gpu:
        preset = "quick"
        reason_key = "hw_gpu_small"
        reason_params = {"mem": f"{gpu_mem_gb:.0f}"}
        reason = f"GPU with {gpu_mem_gb:.0f} GB — small models only"
    elif ram_gb >= 16:
        preset = "standard"
        reason_key = "hw_cpu_standard"
        reason_params = {"mem": f"{ram_gb:.0f}"}
        reason = f"{ram_gb:.0f} GB RAM, CPU — standard models (slow)"
    else:
        preset = "quick"
        reason_key = "hw_cpu_quick"
        reason_params = {"mem": f"{ram_gb:.0f}"}
        reason = f"{ram_gb:.0f} GB RAM, CPU — quick tests only"

    presets = {
        "quick": {
            "d_model": 128, "num_layers": 4, "num_heads": 4, "d_ff": 512,
            "max_seq_len": 128, "vocab_size": 10000,
            "max_iterations": 1000, "batch_size": 8, "learning_rate": 0.001
        },
        "standard": {
            "d_model": 256, "num_layers": 6, "num_heads": 8, "d_ff": 1024,
            "max_seq_len": 256, "vocab_size": 15000,
            "max_iterations": 10000, "batch_size": 16, "learning_rate": 0.0003
        },
        "quality": {
            "d_model": 512, "num_layers": 12, "num_heads": 16, "d_ff": 2048,
            "max_seq_len": 512, "vocab_size": 25000,
            "max_iterations": 100000, "batch_size": 32, "learning_rate": 0.0001
        }
    }

    return {
        "preset": preset,
        "reason": reason,
        "reason_key": reason_key,
        "reason_params": reason_params,
        "config": presets[preset],
        "has_gpu": has_gpu,
        "gpu_memory_gb": round(gpu_mem_gb, 1),
        "ram_gb": round(ram_gb, 1)
    }


# ─────────────────────────────────────────
# GPU — установка и активация из UI
# ─────────────────────────────────────────

CUDA_INDEXES = [
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu121",
]

gpu_install_status = {
    "is_installing": False,
    "progress": 0,
    "message": "",
    "success": False,
    "error": ""
}


def _run_uv(args: list, timeout: int = 60) -> subprocess.CompletedProcess:
    """Запускает uv через текущий Python: python -m uv <args>"""
    return subprocess.run(
        [sys.executable, "-m", "uv"] + args,
        capture_output=True, text=True, timeout=timeout
    )


def _find_cuda_index_for_python(python_exe: str) -> str:
    """Проверяет, какой CUDA-индекс PyTorch поддерживает данный python_exe."""
    for idx in CUDA_INDEXES:
        try:
            r = subprocess.run(
                [python_exe, "-m", "pip", "index", "versions", "torch",
                 "--index-url", idx],
                capture_output=True, text=True, timeout=20
            )
            if r.returncode == 0 and "torch" in r.stdout.lower():
                return idx
        except Exception:
            continue
    return ""


def _get_venv_python(venv_dir: Path) -> str:
    """Путь к python.exe внутри venv (Windows / Linux)."""
    for p in [venv_dir / "Scripts" / "python.exe",
              venv_dir / "bin" / "python"]:
        if p.exists():
            return str(p)
    return ""


@app.get("/gpu_status")
async def get_gpu_status():
    """Полный статус GPU: CUDA, версия Python, наличие GPU-venv."""
    cuda_available = torch.cuda.is_available()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv_ready = bool(_get_venv_python(BASE_DIR / ".venv-gpu"))
    return {
        "cuda_available": cuda_available,
        "pytorch_version": torch.__version__,
        "python_version": py_ver,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "venv_gpu_ready": venv_ready,
        "install_status": gpu_install_status,
    }


@app.post("/gpu_install")
async def install_gpu_pytorch():
    """
    Универсальная установка PyTorch CUDA для любой версии Python.
    Алгоритм:
      1. Если текущий Python поддерживает CUDA-колёса — ставим прямо в него.
      2. Иначе (Python 3.14+): через uv скачиваем Python 3.13, создаём .venv-gpu,
         ставим туда torch+CUDA, создаём start_gpu.bat / start_gpu.sh.
    """
    global gpu_install_status

    if gpu_install_status["is_installing"]:
        return {"status": "already_installing", "message": "Установка уже идёт"}
    if torch.cuda.is_available():
        return {"status": "already_available", "message": "GPU уже доступен!"}

    def _upd(progress: int, message: str):
        gpu_install_status["progress"] = progress
        gpu_install_status["message"] = message

    def do_install():
        global gpu_install_status
        gpu_install_status.update({
            "is_installing": True, "progress": 2,
            "message": "Запускаем...", "success": False, "error": ""
        })

        try:
            # ── Путь А: текущий Python поддерживает CUDA ────────────────────
            _upd(5, "Шаг 1/3: Проверяем CUDA-совместимость текущего Python...")
            cur_idx = _find_cuda_index_for_python(sys.executable)

            if cur_idx:
                cuda_ver = "12.4" if "cu124" in cur_idx else "12.1"
                _upd(15, "Удаляем CPU-версию PyTorch...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall",
                     "torch", "torchvision", "torchaudio", "-y"],
                    capture_output=True, timeout=120
                )
                _upd(30, f"Скачиваем PyTorch CUDA {cuda_ver} (~2 ГБ)...")
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     "torch", "torchvision", "--index-url", cur_idx],
                    capture_output=True, text=True, timeout=900
                )
                if r.returncode == 0:
                    gpu_install_status.update({
                        "progress": 100,
                        "message": "PyTorch CUDA installed! Restart the server.",
                        "success": True
                    })
                else:
                    err = (r.stderr or r.stdout or "")[-600:]
                    gpu_install_status.update({
                        "progress": -1,
                        "message": "pip install error. See error field.",
                        "error": err
                    })
                return

            # ── Путь Б: несовместимый Python (3.14+) → uv → Python 3.13 ────
            _upd(8, "Current Python incompatible with CUDA.\nStep 1/5: Installing uv...")

            uv_check = _run_uv(["--version"], timeout=10)
            if uv_check.returncode != 0:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "uv", "-q"],
                    capture_output=True, timeout=90
                )

            _upd(15, "Step 2/5: Downloading Python 3.13 (~30 MB)...")
            py_target = "3.13"
            r = _run_uv(["python", "install", py_target], timeout=300)
            if r.returncode != 0:
                py_target = "3.12"
                r = _run_uv(["python", "install", py_target], timeout=300)
                if r.returncode != 0:
                    gpu_install_status.update({
                        "progress": -1,
                        "message": f"Failed to download Python {py_target} via uv.\n{r.stderr[-300:]}",
                        "error": r.stderr[-300:]
                    })
                    return

            _upd(35, f"Step 3/5: Creating GPU environment (Python {py_target})...")
            venv_dir = BASE_DIR / ".venv-gpu"
            r = _run_uv(["venv", str(venv_dir), "--python", py_target, "--clear"], timeout=60)
            if r.returncode != 0:
                gpu_install_status.update({
                    "progress": -1,
                    "message": f"Failed to create venv.\n{r.stderr[-300:]}",
                    "error": r.stderr[-300:]
                })
                return

            venv_python = _get_venv_python(venv_dir)
            if not venv_python:
                gpu_install_status.update({
                    "progress": -1,
                    "message": "venv created but python.exe not found.",
                    "error": "venv python not found"
                })
                return

            # Python 3.13 supports cu124; use uv pip (no pip needed in venv)
            idx = CUDA_INDEXES[0]  # cu124
            cuda_ver = "12.4"
            _upd(45, f"Step 4/5: Downloading PyTorch CUDA {cuda_ver} (~2 GB)...")
            r = _run_uv(
                ["pip", "install", "torch", "torchvision",
                 "--python", venv_python,
                 "--index-url", idx],
                timeout=900
            )
            if r.returncode != 0:
                idx = CUDA_INDEXES[1]
                cuda_ver = "12.1"
                _upd(50, f"Step 4/5: cu124 failed, trying CUDA {cuda_ver}...")
                r = _run_uv(
                    ["pip", "install", "torch", "torchvision",
                     "--python", venv_python,
                     "--index-url", idx],
                    timeout=900
                )
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "")[-600:]
                gpu_install_status.update({
                    "progress": -1,
                    "message": "PyTorch install error in venv.",
                    "error": err
                })
                return

            _upd(88, "Step 5/5: Installing project dependencies in venv...")
            _run_uv(
                ["pip", "install", "fastapi", "uvicorn[standard]", "pydantic",
                 "--python", venv_python, "-q"],
                timeout=180
            )

            # Create startup scripts
            _upd(95, "Creating start_gpu.bat / start_gpu.sh...")
            server_path = BASE_DIR / "server_with_datasets.py"

            bat = f'@echo off\necho Starting AZR Trainer with GPU support...\n"{venv_python}" "{server_path}"\npause\n'
            (BASE_DIR / "start_gpu.bat").write_text(bat, encoding="utf-8")

            venv_py_sh = (venv_dir / "bin" / "python")
            sh = f'#!/bin/bash\necho "Starting AZR Trainer with GPU support..."\n"{venv_py_sh}" "{server_path}"\n'
            sh_path = BASE_DIR / "start_gpu.sh"
            sh_path.write_text(sh, encoding="utf-8")
            try:
                sh_path.chmod(0o755)
            except Exception:
                pass

            gpu_install_status.update({
                "progress": 100,
                "message": (
                    "GPU environment ready!\n"
                    "Close the server and restart via start_gpu.bat (Windows) "
                    "or start_gpu.sh (Linux/Mac)."
                ),
                "success": True
            })

        except subprocess.TimeoutExpired:
            gpu_install_status.update({
                "progress": -1,
                "message": "Timeout exceeded. Installation took too long.",
                "error": "timeout"
            })
        except Exception as e:
            gpu_install_status.update({
                "progress": -1,
                "message": f"Error: {e}",
                "error": str(e)
            })
        finally:
            gpu_install_status["is_installing"] = False

    threading.Thread(target=do_install, daemon=True).start()
    return {"status": "installing", "message": "Installation started."}


# ─────────────────────────────────────────
# LLM Autopilot
# ─────────────────────────────────────────

@app.post("/autopilot/start")
async def start_autopilot(config: AutopilotConfig):
    global active_autopilot
    if active_autopilot and active_autopilot.state in ("planning", "executing", "monitoring"):
        raise HTTPException(400, "Autopilot is already running")
    try:
        provider = LLMProvider(config.provider, config.api_key, config.endpoint, config.model)
        executor = ToolExecutor(
            dataset_catalog=dataset_catalog,
            dataset_manager=dataset_manager,
            models_dir=MODELS_DIR,
            books_dir=BOOKS_DIR,
            checkpoints_dir=CHECKPOINTS_DIR,
            training_status=training_status,
            active_models=active_models,
            start_training_fn=train_model_background
        )
        active_autopilot = LLMAutopilot(provider, executor)
        active_autopilot.start(config.goal, time_budget_minutes=config.time_budget or 0)
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(500, f"Failed to start autopilot: {str(e)}")


@app.get("/autopilot/status")
async def get_autopilot_status(since: int = 0):
    if not active_autopilot:
        return {"state": "idle", "log": [], "log_count": 0}
    status = active_autopilot.get_status()
    return {
        "state": status["state"],
        "log": status["log"][since:],
        "log_count": status["log_count"]
    }


@app.post("/autopilot/stop")
async def stop_autopilot():
    if active_autopilot:
        active_autopilot.stop()
        return {"status": "stopping"}
    return {"status": "not_running"}


# ─────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────

if __name__ == "__main__":
    out = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else None
    def _p(s):
        if out:
            out.write((s + "\n").encode("utf-8"))
            out.flush()
        else:
            print(s)
    _p("=" * 60)
    _p("  AZR Model Trainer v2 + LLM Autopilot")
    _p("  Features: Catalog, Analytics, REINFORCE, Comparison")
    _p("=" * 60)
    _p(f"Models: {MODELS_DIR}")
    _p(f"Books: {BOOKS_DIR}")
    _p(f"Checkpoints: {CHECKPOINTS_DIR}")
    _p(f"Reports: {REPORTS_DIR}")
    _p(f"Catalog: {len(dataset_catalog.catalog)} datasets")
    cuda = torch.cuda.is_available()
    _p(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if cuda else 'CPU'}")
    if not cuda:
        _p("")
        _p("  ! GPU not found. Open browser and click 'Activate GPU'")
    _p("=" * 60)
    _p("")
    _p("  >>> Open in browser: http://localhost:8000 <<<")
    _p("")
    uvicorn.run(app, host="0.0.0.0", port=8000)
