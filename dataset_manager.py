"""
Система управления датасетами для моделей
Позволяет:
- Прикреплять несколько датасетов к модели
- Автоматически загружать прикреплённые при обучении
- Открепять датасеты (они возвращаются в общий список)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class DatasetManager:
    def __init__(self, db_path="datasets_db.json"):
        self.db_path = Path(db_path)
        self.db = self._load_db()
    
    def _load_db(self) -> Dict:
        """Загрузить базу данных связей"""
        if self.db_path.exists():
            # utf-8-sig: переживает BOM (его добавляют Блокнот и PowerShell)
            try:
                with open(self.db_path, 'r', encoding='utf-8-sig') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"WARNING: datasets_db.json is corrupted ({e}); starting with an empty database")
        return {
            "models": {},  # model_name -> {"datasets": [...], "created": "..."}
            "datasets": {}  # dataset_name -> {"path": "...", "size": 123, "attached_to": [...]}
        }
    
    def _save_db(self):
        """Сохранить базу данных"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, indent=2, ensure_ascii=False)
    
    def register_dataset(self, dataset_name: str, dataset_path: str,
                         metadata: dict = None) -> bool:
        """Зарегистрировать датасет в системе"""
        path = Path(dataset_path)
        if not path.exists():
            return False

        if dataset_name not in self.db["datasets"]:
            entry = {
                "path": str(path),
                "size": path.stat().st_size,
                "attached_to": [],
                "registered": datetime.now().isoformat()
            }
            # Дополнительные метаданные (формат, кодировка, источник и т.д.)
            if metadata:
                entry["format"] = metadata.get("format", "txt")
                entry["encoding"] = metadata.get("encoding", "utf-8")
                entry["source"] = metadata.get("source", "upload")
                entry["lines"] = metadata.get("lines", 0)
                entry["description"] = metadata.get("description", "")
                # Человекочитаемое название («Анна Каренина»), файл остаётся техническим
                if metadata.get("title"):
                    entry["title"] = metadata["title"]
                if metadata.get("catalog_id"):
                    entry["catalog_id"] = metadata["catalog_id"]
            self.db["datasets"][dataset_name] = entry
            self._save_db()

        return True
    
    def attach_dataset(self, model_name: str, dataset_name: str) -> bool:
        """Прикрепить датасет к модели"""
        # Создать запись модели если нет
        if model_name not in self.db["models"]:
            self.db["models"][model_name] = {
                "datasets": [],
                "created": datetime.now().isoformat()
            }
        
        # Проверить что датасет существует
        if dataset_name not in self.db["datasets"]:
            return False
        
        # Прикрепить если ещё не прикреплён
        if dataset_name not in self.db["models"][model_name]["datasets"]:
            self.db["models"][model_name]["datasets"].append(dataset_name)
            self.db["datasets"][dataset_name]["attached_to"].append(model_name)
            self._save_db()
        
        return True
    
    def detach_dataset(self, model_name: str, dataset_name: str) -> bool:
        """Открепить датасет от модели"""
        if model_name not in self.db["models"]:
            return False
        
        # Удалить из модели
        if dataset_name in self.db["models"][model_name]["datasets"]:
            self.db["models"][model_name]["datasets"].remove(dataset_name)
        
        # Удалить из датасета
        if dataset_name in self.db["datasets"]:
            if model_name in self.db["datasets"][dataset_name]["attached_to"]:
                self.db["datasets"][dataset_name]["attached_to"].remove(model_name)
        
        self._save_db()
        return True
    
    def get_attached_datasets(self, model_name: str) -> List[str]:
        """Получить список прикреплённых датасетов"""
        if model_name not in self.db["models"]:
            return []
        return self.db["models"][model_name]["datasets"]
    
    def get_available_datasets(self, model_name: str = None) -> List[Dict]:
        """Получить список доступных датасетов (не прикреплённых к модели)"""
        if model_name is None:
            # Все датасеты
            return [
                {
                    "name": name,
                    "title": info.get("title") or name,
                    "path": info["path"],
                    "size": info["size"],
                    "attached_to": info["attached_to"]
                }
                for name, info in self.db["datasets"].items()
            ]

        # Только не прикреплённые к этой модели
        attached = self.get_attached_datasets(model_name)
        return [
            {
                "name": name,
                "title": info.get("title") or name,
                "path": info["path"],
                "size": info["size"],
                "attached_to": info["attached_to"]
            }
            for name, info in self.db["datasets"].items()
            if name not in attached
        ]
    
    def get_dataset_paths(self, model_name: str) -> List[str]:
        """Получить пути к прикреплённым датасетам"""
        dataset_names = self.get_attached_datasets(model_name)
        paths = []
        
        for name in dataset_names:
            if name in self.db["datasets"]:
                path = self.db["datasets"][name]["path"]
                if Path(path).exists():
                    paths.append(path)
        
        return paths
    
    def load_attached_texts(self, model_name: str) -> List[str]:
        """Загрузить тексты из всех прикреплённых датасетов"""
        paths = self.get_dataset_paths(model_name)
        texts = []
        
        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Разбиваем на чанки
                    chunks = [text[i:i+1000] for i in range(0, len(text), 500)]
                    texts.extend(chunks)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        return texts
    
    def get_model_info(self, model_name: str) -> Dict:
        """Получить полную информацию о модели"""
        if model_name not in self.db["models"]:
            return None
        
        datasets = self.get_attached_datasets(model_name)
        dataset_info = []
        total_size = 0
        
        for ds_name in datasets:
            if ds_name in self.db["datasets"]:
                info = self.db["datasets"][ds_name]
                dataset_info.append({
                    "name": ds_name,
                    "title": info.get("title") or ds_name,
                    "size": info["size"],
                    "path": info["path"]
                })
                total_size += info["size"]
        
        return {
            "model_name": model_name,
            "datasets": dataset_info,
            "total_datasets": len(datasets),
            "total_size": total_size,
            "created": self.db["models"][model_name].get("created")
        }
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Удалить датасет из системы (открепить от всех моделей)"""
        if dataset_name not in self.db["datasets"]:
            return False
        
        # Открепить от всех моделей
        attached_models = self.db["datasets"][dataset_name]["attached_to"].copy()
        for model_name in attached_models:
            self.detach_dataset(model_name, dataset_name)
        
        # Удалить из БД
        del self.db["datasets"][dataset_name]
        self._save_db()
        
        return True
    
    def get_stats(self) -> Dict:
        """Статистика системы"""
        return {
            "total_models": len(self.db["models"]),
            "total_datasets": len(self.db["datasets"]),
            "attachments": sum(
                len(model_info["datasets"])
                for model_info in self.db["models"].values()
            )
        }


# Пример использования
if __name__ == "__main__":
    print("="*60)
    print("DEMO: Dataset Manager")
    print("="*60)
    
    manager = DatasetManager("demo_db.json")
    
    # Регистрация датасетов
    print("\n1. Регистрация датасетов:")
    manager.register_dataset("book1", "books/example_book.txt")
    manager.register_dataset("book2", "books/book2.txt")
    manager.register_dataset("book3", "books/book3.txt")
    print("   ✓ Зарегистрировано 3 датасета")
    
    # Прикрепление к модели
    print("\n2. Прикрепление к модели 'my_model':")
    manager.attach_dataset("my_model", "book1")
    manager.attach_dataset("my_model", "book2")
    print("   ✓ Прикреплено 2 датасета")
    
    # Список прикреплённых
    print("\n3. Прикреплённые датасеты:")
    attached = manager.get_attached_datasets("my_model")
    print(f"   {attached}")
    
    # Список доступных
    print("\n4. Доступные датасеты (не прикреплённые):")
    available = manager.get_available_datasets("my_model")
    for ds in available:
        print(f"   - {ds['name']}")
    
    # Открепление
    print("\n5. Открепление 'book1':")
    manager.detach_dataset("my_model", "book1")
    print(f"   ✓ Теперь прикреплено: {manager.get_attached_datasets('my_model')}")
    
    # Информация
    print("\n6. Информация о модели:")
    info = manager.get_model_info("my_model")
    if info:
        print(f"   Датасетов: {info['total_datasets']}")
        print(f"   Общий размер: {info['total_size']} bytes")
    
    # Статистика
    print("\n7. Общая статистика:")
    stats = manager.get_stats()
    print(f"   Моделей: {stats['total_models']}")
    print(f"   Датасетов: {stats['total_datasets']}")
    print(f"   Связей: {stats['attachments']}")
    
    print("\n" + "="*60)
