"""
AZR Trainer с поддержкой:
- REINFORCE policy gradient (self-play влияет на веса модели)
- Многокомпонентная система наград
- Детальная аналитика обучения
- Пауза/возобновление
- Расширенные callback'и статуса
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
import time
import math
from datetime import datetime
import random

from reward_model import RewardComputer
from training_analytics import TrainingAnalytics


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) == max_length:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class AZRTrainer:
    def __init__(self, model, tokenizer,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 status_callback=None,
                 reward_computer=None,
                 analytics=None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.training_history = []
        self.iteration = 0
        self.status_callback = status_callback
        self.current_loss = 0.0
        self.current_reward = 0.0
        self.current_reward_components = {}
        self.current_perplexity = 0.0
        self.optimizer = None
        self.scheduler = None
        self.should_stop = False

        # Новые компоненты
        self.reward_computer = reward_computer or RewardComputer(tokenizer)
        self.analytics = analytics
        self.eval_data = []  # Валидационные данные для перплексии

        # Метрики скорости
        self._tokens_per_sec = 0.0
        self._eta_seconds = -1
        self._memory_mb = 0.0

        # Параметры по-батчевого сохранения (устанавливаются в train_continuous)
        self._save_every = 0
        self._checkpoint_dir = Path('checkpoints')

    def train_step(self, batch):
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        logits, loss = self.model(x, y)

        return loss

    def self_play_step(self, prompts, max_length=50, temperature=0.8):
        self.model.eval()
        generated_texts = []

        with torch.no_grad():
            for prompt in prompts:
                tokens = self.tokenizer.encode(prompt)
                if len(tokens) == 0:
                    continue
                idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

                generated = self.model.generate(idx, max_new_tokens=max_length,
                                               temperature=temperature, top_k=40)

                generated_tokens = generated[0].cpu().tolist()
                generated_text = self.tokenizer.decode(generated_tokens)
                generated_texts.append(generated_text)

        return generated_texts

    def compute_reward(self, generated_texts):
        """Многокомпонентная награда через RewardComputer"""
        detailed_rewards = self.reward_computer.compute_batch_reward(generated_texts)

        totals = np.array([r["total"] for r in detailed_rewards])

        # Сохраняем компоненты последней оценки для статуса
        if detailed_rewards:
            self.current_reward_components = detailed_rewards[-1].get("components", {})

        return totals, detailed_rewards

    def _reinforce_step(self, generated_texts, rewards, optimizer):
        """REINFORCE policy gradient: лёгкая коррекция на основе self-play наград"""
        self.model.train()
        baseline = float(np.mean(rewards))

        for text, reward in zip(generated_texts, rewards):
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 3:
                continue

            advantage = reward - baseline
            # Пропускаем только при совсем нулевом advantage
            if abs(advantage) < 0.001:
                continue

            # Обрезаем если слишком длинный
            max_len = min(len(tokens), self.model.max_seq_len)
            tokens = tokens[:max_len]

            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)

            try:
                logits, loss = self.model(x, y)
                if loss is None:
                    continue

                # Policy gradient: лёгкая коррекция (0.005 — мягче чтобы не мешать основному обучению)
                pg_loss = -advantage * 0.005 * loss

                optimizer.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
            except Exception:
                continue

    @torch.no_grad()
    def _compute_val_loss(self) -> float:
        """Validation loss на отложенной выборке"""
        if not hasattr(self, '_val_loader'):
            return 0.0
        self.model.eval()
        total, n = 0.0, 0
        for batch in self._val_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            if loss is not None:
                total += loss.item()
                n += 1
        return total / max(n, 1)

    def _get_memory_mb(self) -> float:
        """Получить использование памяти"""
        try:
            if torch.cuda.is_available() and self.device.type != 'cpu':
                return torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                try:
                    import psutil
                    return psutil.Process().memory_info().rss / (1024 * 1024)
                except ImportError:
                    return 0.0
        except Exception:
            return 0.0

    def azr_train_epoch(self, dataloader, optimizer, num_self_play=5, max_iterations=0):
        epoch_loss = 0
        epoch_reward = 0
        num_batches = 0
        reward_count = 0

        sample_texts = [
            "Искусственный интеллект",
            "В далёком будущем",
            "Once upon a time",
            "The future of AI",
            "In a distant galaxy",
            "Секрет успеха заключается в",
            "Technology has changed",
            "Однажды в тёмном лесу",
        ]

        for batch_idx, batch in enumerate(dataloader):
            if self.should_stop:
                print("Training stopped by user")
                break

            batch_start = time.perf_counter()

            loss = self.train_step(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            self.current_loss = loss.item()

            # Замер скорости
            batch_time = time.perf_counter() - batch_start
            batch_tokens = batch[0].shape[0] * batch[0].shape[1]
            if batch_time > 0:
                self._tokens_per_sec = batch_tokens / batch_time

            if self.analytics:
                self.analytics.record_batch_time(batch_tokens)

            # Self-play каждые 50 батчей (реже = быстрее обучение)
            if batch_idx % 50 == 0:
                try:
                    prompts = random.sample(sample_texts, min(num_self_play, len(sample_texts)))
                    generated = self.self_play_step(prompts, max_length=30)
                    if len(generated) > 0:
                        rewards, detailed = self.compute_reward(generated)
                        self.current_reward = float(rewards.mean())
                        epoch_reward += self.current_reward
                        reward_count += 1
                except Exception as e:
                    print(f"Self-play error: {e}")
                    self.current_reward = 0.0

            # REINFORCE каждые 100 батчей (лёгкая коррекция, не основное обучение)
            if batch_idx % 100 == 0 and batch_idx > 0:
                try:
                    prompts = random.sample(sample_texts, min(3, len(sample_texts)))
                    generated = self.self_play_step(prompts, max_length=40)
                    if len(generated) > 0:
                        rewards, _ = self.compute_reward(generated)
                        self._reinforce_step(generated, rewards, optimizer)
                except Exception as e:
                    print(f"REINFORCE error: {e}")

            num_batches += 1
            self.iteration += 1

            # Чекпоинт СТРОГО каждые save_every итераций (раньше сохранялся только
            # на границе эпохи и лишь при случайном совпадении с save_every —
            # чекпоинты появлялись непредсказуемо)
            if self._save_every > 0 and self.iteration % self._save_every == 0:
                self.save_checkpoint(
                    self._checkpoint_dir / f"model_iter_{self.iteration}.pt",
                    save_optimizer=True
                )
                self.cleanup_old_checkpoints(self._checkpoint_dir, keep_last=7)
                if self.analytics:
                    self._run_checkpoint_analytics(self.current_loss, self.current_reward)

            # Останавливаемся ровно на max_iterations
            if max_iterations > 0 and self.iteration >= max_iterations:
                break

            # Обновление памяти
            self._memory_mb = self._get_memory_mb()

            # Расширенный status callback каждые 20 батчей
            if self.status_callback and batch_idx % 20 == 0:
                try:
                    eta = -1
                    if self.analytics:
                        eta = self.analytics.get_eta(self.iteration, max_iterations)

                    self.status_callback({
                        'current_iteration': self.iteration,
                        'max_iterations': max_iterations,
                        'current_loss': float(self.current_loss),
                        'current_reward': float(self.current_reward),
                        'is_training': True,
                        'phase': 'training',
                        'perplexity': float(self.current_perplexity),
                        'tokens_per_sec': round(self._tokens_per_sec, 1),
                        'eta_seconds': eta,
                        'reward_components': dict(self.current_reward_components),
                        'memory_mb': round(self._memory_mb, 1),
                    })
                except Exception as e:
                    print(f"Status callback error: {e}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_reward = epoch_reward / max(reward_count, 1)

        return avg_loss, avg_reward

    def train_continuous(self, texts, max_iterations=1000000, batch_size=16, lr=3e-4,
                        save_every=1000, checkpoint_dir='checkpoints', resume_from=None):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        # Для по-батчевого сохранения внутри azr_train_epoch
        self._save_every = max(0, int(save_every))
        self._checkpoint_dir = checkpoint_dir

        # Chunk length must not exceed the model's max_seq_len (crash otherwise)
        chunk_len = min(128, self.model.max_seq_len)
        dataset = TextDataset(texts, self.tokenizer, max_length=chunk_len)
        if len(dataset) == 0:
            print("ERROR: No samples in dataset! Check your text data.")
            return []

        # Train / validation split (90/10)
        val_size = max(1, int(len(dataset) * 0.10))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.eval_data = [dataset.samples[i] for i in range(min(val_size, len(dataset.samples)))]
        self._val_dataset = val_dataset

        use_pin = (self.device.type == 'cuda')
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=use_pin)
        self._val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=0, pin_memory=use_pin)

        # Обновляем RewardComputer референсными данными
        self.reward_computer.update_reference(texts[:100])

        # Always create optimizer/scheduler first
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_iterations)

        # Resume from checkpoint if provided
        if resume_from and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = self.load_checkpoint(resume_from, load_optimizer=True)
            print(f"Resumed from iteration {self.iteration}")

        # Запускаем аналитику
        if self.analytics:
            self.analytics.start_session()

        print(f"Starting AZR training with {len(dataset)} samples")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Starting from iteration: {self.iteration}")
        print(f"Target iterations: {max_iterations}")
        print(f"REINFORCE: enabled (every 20 batches)")
        print(f"Reward components: {list(RewardComputer.WEIGHTS.keys())}")

        epoch = 0
        self.should_stop = False

        try:
            while self.iteration < max_iterations and not self.should_stop:
                epoch += 1
                avg_loss, avg_reward = self.azr_train_epoch(
                    dataloader, self.optimizer, max_iterations=max_iterations
                )

                if self.should_stop:
                    break

                self.scheduler.step()

                val_loss = self._compute_val_loss()
                overfit = " ⚠ OVERFIT" if val_loss > avg_loss * 1.1 else ""

                self.training_history.append({
                    'epoch': epoch,
                    'iteration': self.iteration,
                    'loss': avg_loss,
                    'val_loss': val_loss,
                    'reward': avg_reward,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now().isoformat()
                })

                print(f"Epoch {epoch} | Iter {self.iteration}/{max_iterations} | "
                      f"Loss: {avg_loss:.4f} | Val: {val_loss:.4f}{overfit} | "
                      f"Reward: {avg_reward:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Speed: {self._tokens_per_sec:.0f} tok/s")

                # Чекпоинты сохраняются по-батчево внутри azr_train_epoch —
                # строго каждые save_every итераций

            if self.should_stop:
                print("Training paused. Save checkpoint to resume later.")
                self.save_checkpoint(
                    checkpoint_dir / f"model_paused_{self.iteration}.pt",
                    save_optimizer=True, kind="paused"
                )
            else:
                print("Training completed!")

            # Финальный отчёт аналитики
            if self.analytics:
                self.analytics.save_report()

            if self.status_callback:
                self.status_callback({
                    'current_iteration': self.iteration,
                    'max_iterations': max_iterations,
                    'current_loss': float(self.current_loss),
                    'current_reward': float(self.current_reward),
                    'is_training': False,
                    'phase': 'idle',
                    'perplexity': float(self.current_perplexity),
                    'tokens_per_sec': 0,
                    'eta_seconds': 0,
                    'reward_components': dict(self.current_reward_components),
                    'memory_mb': round(self._memory_mb, 1),
                })

        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C)")
            print("Saving checkpoint...")
            self.save_checkpoint(
                checkpoint_dir / f"model_interrupted_{self.iteration}.pt",
                save_optimizer=True, kind="interrupted"
            )
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            print("Saving checkpoint...")
            self.save_checkpoint(
                checkpoint_dir / f"model_error_{self.iteration}.pt",
                save_optimizer=True, kind="error"
            )

        return self.training_history

    def _run_checkpoint_analytics(self, avg_loss, avg_reward):
        """Запустить полную аналитику на чекпоинте"""
        try:
            # Перплексия
            perplexity = self.analytics.compute_perplexity(
                self.model, self.eval_data, self.device
            )
            self.current_perplexity = perplexity

            # Бенчмарки
            benchmarks = self.analytics.run_benchmarks(
                self.model, self.tokenizer, self.device,
                reward_computer=self.reward_computer
            )
            self.analytics.record_benchmarks(self.iteration, benchmarks)

            # Использование словаря
            sample_texts = [b["text"] for b in benchmarks if "text" in b]
            vocab_usage = self.analytics.compute_vocab_usage(sample_texts, self.tokenizer)

            # Полный отчёт по итерации
            reward_breakdown = {
                "total": avg_reward,
                "components": dict(self.current_reward_components)
            }

            self.analytics.record_iteration(
                iteration=self.iteration,
                loss=avg_loss,
                reward_breakdown=reward_breakdown,
                generated_samples=sample_texts[:3],
                perplexity=perplexity,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                tokens_per_sec=self._tokens_per_sec,
                vocab_usage=vocab_usage
            )

            print(f"   Analytics: perplexity={perplexity:.1f}, "
                  f"vocab_coverage={vocab_usage.get('coverage_pct', 0):.1f}%")

        except Exception as e:
            print(f"Analytics error: {e}")

    def cleanup_old_checkpoints(self, checkpoint_dir, keep_last=5):
        try:
            import glob
            pattern = str(checkpoint_dir / "model_iter_*.pt")
            checkpoints = sorted(glob.glob(pattern), key=lambda x: Path(x).stat().st_mtime)

            if len(checkpoints) > keep_last:
                to_delete = checkpoints[:-keep_last]
                deleted_names = []
                for cp in to_delete:
                    try:
                        Path(cp).unlink()
                        deleted_names.append(Path(cp).name)
                        print(f"   Deleted old checkpoint: {Path(cp).name}")
                    except Exception as e:
                        print(f"   Failed to delete {Path(cp).name}: {e}")
                # Убираем удалённые файлы из манифеста
                if deleted_names:
                    try:
                        manifest_path = Path(checkpoint_dir) / "manifest.json"
                        if manifest_path.exists():
                            with open(manifest_path, "r", encoding="utf-8") as f:
                                manifest = json.load(f)
                            for name in deleted_names:
                                manifest.pop(name, None)
                            with open(manifest_path, "w", encoding="utf-8") as f:
                                json.dump(manifest, f, ensure_ascii=False, indent=1)
                    except Exception as e:
                        print(f"   Manifest prune failed: {e}")
        except Exception as e:
            print(f"   Cleanup failed: {e}")

    def stop_training(self):
        self.should_stop = True
        print("Stop signal received, will pause after current batch...")

    def _update_manifest(self, path, kind):
        """Записать метаданные чекпоинта в manifest.json рядом с файлами.
        Благодаря манифесту UI показывает loss/дату/тип без загрузки .pt файлов."""
        try:
            path = Path(path)
            manifest_path = path.parent / "manifest.json"
            manifest = {}
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except (json.JSONDecodeError, OSError):
                    manifest = {}
            val_loss = 0.0
            if self.training_history:
                val_loss = self.training_history[-1].get('val_loss', 0.0)
            manifest[path.name] = {
                "iteration": self.iteration,
                "kind": kind,  # auto | paused | interrupted | error
                "loss": round(float(self.current_loss), 4),
                "val_loss": round(float(val_loss), 4),
                "reward": round(float(self.current_reward), 4),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            tmp = manifest_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=1)
            tmp.replace(manifest_path)
        except Exception as e:
            print(f"   Manifest update failed: {e}")

    def save_checkpoint(self, path, save_optimizer=False, kind="auto"):
        try:
            # Full architecture config so a checkpoint alone is enough to rebuild the model
            try:
                first_block = self.model.blocks[0]
                arch_extra = {
                    'num_layers': len(self.model.blocks),
                    'num_heads': first_block.attention.num_heads,
                    'd_ff': first_block.ff.linear1.out_features,
                }
            except Exception:
                arch_extra = {}
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'iteration': self.iteration,
                'training_history': self.training_history[-100:],
                'model_config': {
                    'vocab_size': self.model.vocab_size,
                    'd_model': self.model.d_model,
                    'max_seq_len': self.model.max_seq_len,
                    **arch_extra
                },
                'timestamp': datetime.now().isoformat()
            }

            # Сохраняем токенизатор в чекпоинт для портабельности
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                checkpoint['tokenizer_data'] = {
                    'token_to_id': self.tokenizer.token_to_id,
                    'id_to_token': {str(k): v for k, v in self.tokenizer.id_to_token.items()},
                    'vocab_size': self.tokenizer.vocab_size
                }

            if save_optimizer and self.optimizer is not None:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
            self._update_manifest(path, kind)
            print(f"Checkpoint saved: {path.name}")
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            try:
                fallback_path = str(path).replace('.pt', '_state.pt')
                torch.save(self.model.state_dict(), fallback_path)
                print(f"Model state saved as fallback: {Path(fallback_path).name}")
                return True
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return False

    def load_checkpoint(self, path, load_optimizer=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.iteration = checkpoint.get('iteration', 0)
        self.training_history = checkpoint.get('training_history', [])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state restored")

            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state restored")

        print(f"Checkpoint loaded from iteration {self.iteration}")
        return checkpoint
