#!/usr/bin/env python3
"""
Строит ПОЛНЫЙ интерфейс AZR Model Trainer v2
- Каталог датасетов с поиском
- Детальная аналитика обучения
- Сравнение итераций
- Визуализация confidence токенов
- Wizard для новичков
- Пресеты конфигурации
- Milestone-прогресс
- "Что сейчас происходит"
"""

import os
from pathlib import Path

# Определяем путь к templates
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

html = r'''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AZR Model Trainer v2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); min-height: 100vh; padding: 15px; color: #e0e0e0; }
        .container { max-width: 1500px; margin: 0 auto; background: #1a1a2e; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.5); border: 1px solid #333; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; text-align: center; border-radius: 16px 16px 0 0; position: relative; }
        .header h1 { font-size: 2em; margin-bottom: 5px; }
        .header p { opacity: 0.9; font-size: 0.95em; }
        .lang-toggle { position: absolute; top: 15px; right: 20px; display: flex; gap: 0; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.3); }
        .lang-btn { padding: 6px 14px; background: rgba(255,255,255,0.1); color: rgba(255,255,255,0.7); border: none; cursor: pointer; font-weight: 600; font-size: 0.85em; transition: all 0.3s; }
        .lang-btn.active { background: rgba(255,255,255,0.3); color: white; }
        .lang-btn:hover { background: rgba(255,255,255,0.2); }
        .tabs { display: flex; background: #16213e; flex-wrap: wrap; border-bottom: 1px solid #333; overflow-x: auto; }
        .tab { flex: 1; min-width: 100px; padding: 12px 8px; text-align: center; cursor: pointer; background: #16213e; border: none; font-size: 0.85em; font-weight: 600; color: #888; transition: all 0.3s; white-space: nowrap; }
        .tab:hover { background: #1a1a2e; color: #aaa; }
        .tab.active { background: #1a1a2e; color: #667eea; border-bottom: 2px solid #667eea; }
        .tab-content { display: none; padding: 25px; max-height: 85vh; overflow-y: auto; }
        .tab-content.active { display: block; }
        .form-group { margin-bottom: 18px; position: relative; }
        .form-group label { display: flex; align-items: center; gap: 6px; font-weight: 600; margin-bottom: 6px; color: #ccc; font-size: 0.9em; }
        .help-icon { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; background: #667eea; color: white; font-size: 10px; font-weight: bold; cursor: help; position: relative; }
        .tooltip { display: none; position: absolute; background: #0a0a1a; color: #ddd; padding: 10px; border-radius: 8px; font-size: 11px; width: 280px; z-index: 1000; left: 100%; top: -10px; margin-left: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); line-height: 1.4; font-weight: normal; border: 1px solid #444; }
        .help-icon:hover .tooltip { display: block; }
        .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 10px; border: 1px solid #444; border-radius: 8px; font-size: 0.95em; background: #16213e; color: #e0e0e0; transition: border-color 0.3s; }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus { outline: none; border-color: #667eea; }
        .recommended { font-size: 0.8em; color: #10b981; margin-top: 3px; }
        .btn { padding: 10px 24px; border: none; border-radius: 8px; font-size: 0.9em; font-weight: 600; cursor: pointer; transition: all 0.3s; margin-right: 8px; margin-bottom: 8px; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
        .btn-success { background: #10b981; color: white; }
        .btn-success:hover { background: #059669; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-danger:hover { background: #dc2626; }
        .btn-outline { background: transparent; color: #667eea; border: 1px solid #667eea; }
        .btn-outline:hover { background: #667eea22; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; }
        .chart-container { position: relative; height: 300px; margin: 15px 0; background: #16213e; border-radius: 12px; padding: 15px; border: 1px solid #333; }
        .ap-log-entry { padding: 6px 0; border-bottom: 1px solid #222; line-height: 1.6; }
        .ap-time { color: #555; font-family: monospace; font-size: 0.8em; margin-right: 6px; }
        .ap-type { font-weight: 600; padding: 1px 6px; border-radius: 4px; font-size: 0.75em; margin-right: 6px; display: inline-block; min-width: 60px; text-align: center; }
        .ap-log-thinking .ap-type { background: #667eea33; color: #667eea; }
        .ap-log-assistant .ap-type { background: #10b98133; color: #10b981; }
        .ap-log-tool_call .ap-type { background: #f59e0b33; color: #f59e0b; }
        .ap-log-tool_result .ap-type { background: #06b6d433; color: #06b6d4; }
        .ap-log-status .ap-type { background: #8b5cf633; color: #8b5cf6; }
        .ap-log-system .ap-type { background: #ef444433; color: #ef4444; }
        .ap-log-error .ap-type { background: #ef444433; color: #ef4444; }
        .ap-content { color: #ccc; word-break: break-word; }
        .status-box { background: #16213e; border: 1px solid #333; border-radius: 12px; padding: 18px; margin-top: 15px; }
        .status-box h3 { color: #667eea; margin-bottom: 12px; font-size: 1em; }
        .status-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #222; }
        .status-label { font-weight: 600; color: #888; font-size: 0.9em; }
        .status-value { color: #e0e0e0; font-weight: 600; font-size: 0.9em; }
        .progress-bar { width: 100%; height: 24px; background: #222; border-radius: 12px; overflow: hidden; margin: 12px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.8em; }
        .info-card { background: #16213e; border: 1px solid #333; border-radius: 12px; padding: 18px; margin-bottom: 12px; }
        .info-card h3 { color: #667eea; margin-bottom: 8px; font-size: 1em; }
        .info-card h4 { color: #ccc; margin: 12px 0 6px; font-size: 0.95em; }
        .info-card ul, .info-card ol { margin-left: 18px; line-height: 1.7; color: #aaa; font-size: 0.9em; }
        .alert { padding: 12px; border-radius: 8px; margin-bottom: 15px; font-size: 0.9em; }
        .alert-success { background: #10b98122; color: #10b981; border: 1px solid #10b981; }
        .alert-error { background: #ef444422; color: #ef4444; border: 1px solid #ef4444; }
        .alert-info { background: #667eea22; color: #667eea; border: 1px solid #667eea; }
        table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
        table th, table td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        table th { color: #667eea; font-weight: 600; }
        table td { color: #ccc; }
        code { background: #0a0a1a; padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #667eea; font-size: 0.85em; }
        .dataset-item { background: #0f0f23; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; margin: 6px 0; display: flex; justify-content: space-between; align-items: center; font-size: 0.9em; }
        .catalog-card { background: #16213e; border: 1px solid #333; border-radius: 12px; padding: 16px; transition: all 0.3s; cursor: default; }
        .catalog-card:hover { border-color: #667eea; transform: translateY(-2px); }
        .catalog-card h4 { color: #e0e0e0; margin-bottom: 4px; font-size: 0.95em; }
        .catalog-card .desc { color: #888; font-size: 0.8em; line-height: 1.4; margin: 6px 0; }
        .catalog-card .badges { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }
        .badge { padding: 2px 8px; border-radius: 10px; font-size: 0.7em; font-weight: 600; }
        .badge-lang { background: #667eea33; color: #667eea; }
        .badge-size { background: #10b98133; color: #10b981; }
        .badge-cat { background: #f59e0b33; color: #f59e0b; }
        .badge-downloaded { background: #10b98133; color: #10b981; }
        .preset-card { background: #0f0f23; border: 2px solid #333; border-radius: 12px; padding: 20px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .preset-card:hover { border-color: #667eea; transform: translateY(-2px); }
        .preset-card.selected { border-color: #667eea; background: #667eea11; }
        .preset-card h4 { color: #e0e0e0; margin-bottom: 6px; }
        .preset-card .preset-desc { color: #888; font-size: 0.8em; }
        .preset-card .preset-specs { color: #667eea; font-size: 0.75em; margin-top: 6px; font-family: monospace; }
        .token-viz { line-height: 2; padding: 12px; }
        .token { display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 3px; font-size: 0.9em; cursor: help; position: relative; }
        .token:hover::after { content: attr(data-info); position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: #0a0a1a; color: #ddd; padding: 6px 10px; border-radius: 6px; font-size: 0.75em; white-space: nowrap; z-index: 100; border: 1px solid #444; }
        .milestones { display: flex; justify-content: space-between; padding: 10px 0; margin: 10px 0; }
        .milestone { text-align: center; flex: 1; position: relative; }
        .milestone::before { content: ''; position: absolute; top: 12px; left: 0; right: 0; height: 2px; background: #333; z-index: 0; }
        .milestone.reached::before { background: #667eea; }
        .milestone-dot { width: 24px; height: 24px; border-radius: 50%; background: #333; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; position: relative; z-index: 1; margin-bottom: 4px; }
        .milestone.reached .milestone-dot { background: #667eea; color: white; }
        .milestone-label { font-size: 0.65em; color: #666; display: block; }
        .milestone.reached .milestone-label { color: #667eea; }
        .whats-happening { background: linear-gradient(135deg, #667eea11, #764ba211); border: 1px solid #667eea44; border-radius: 12px; padding: 16px; margin: 12px 0; display: none; }
        .whats-happening h4 { color: #667eea; margin-bottom: 8px; font-size: 0.9em; }
        .whats-happening p { color: #aaa; font-size: 0.85em; line-height: 1.5; }
        .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .compare-panel { background: #0f0f23; border: 1px solid #333; border-radius: 12px; padding: 15px; }
        .compare-panel h4 { color: #667eea; margin-bottom: 8px; font-size: 0.9em; }
        .compare-panel .text-output { color: #ccc; font-size: 0.85em; line-height: 1.6; min-height: 100px; }
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); z-index: 9999; align-items: center; justify-content: center; }
        .modal-overlay.active { display: flex; }
        .modal { background: #1a1a2e; border: 1px solid #444; border-radius: 16px; padding: 30px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; }
        .modal h2 { color: #667eea; margin-bottom: 15px; }
        .modal p { color: #aaa; line-height: 1.6; margin-bottom: 12px; }
        .category-filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 15px; }
        .cat-btn { padding: 6px 14px; border-radius: 20px; border: 1px solid #444; background: transparent; color: #aaa; cursor: pointer; font-size: 0.8em; transition: all 0.2s; }
        .cat-btn:hover, .cat-btn.active { border-color: #667eea; color: #667eea; background: #667eea11; }
        .radar-container { height: 250px; }
        .search-box { display: flex; gap: 8px; margin-bottom: 15px; }
        .search-box input { flex: 1; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f0f23; }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #667eea; }
        @media (max-width: 768px) {
            .compare-grid { grid-template-columns: 1fr; }
            .grid-3 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AZR Model Trainer v2</h1>
            <p data-i18n="subtitle">Создавайте, обучайте и анализируйте нейросети — просто и наглядно</p>
            <div class="lang-toggle">
                <button class="lang-btn active" onclick="switchLanguage('ru')" id="lang_ru">RU</button>
                <button class="lang-btn" onclick="switchLanguage('en')" id="lang_en">EN</button>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('help')" data-i18n="tab_help">Помощь</button>
            <button class="tab" onclick="showTab('catalog')" data-i18n="tab_catalog">Каталог</button>
            <button class="tab" onclick="showTab('create')" data-i18n="tab_create">Создать</button>
            <button class="tab" onclick="showTab('datasets')" data-i18n="tab_datasets">Датасеты</button>
            <button class="tab" onclick="showTab('train')" data-i18n="tab_train">Обучение</button>
            <button class="tab" onclick="showTab('generate')" data-i18n="tab_generate">Генерация</button>
            <button class="tab" onclick="showTab('compare')" data-i18n="tab_compare">Сравнение</button>
            <button class="tab" onclick="showTab('models')" data-i18n="tab_models">Модели</button>
            <button class="tab" onclick="showTab('autopilot')" style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;position:relative;"><span data-i18n="tab_autopilot">Автопилот</span> <span style="font-size:0.6em;background:#f59e0b;color:#000;padding:1px 5px;border-radius:8px;font-weight:700;vertical-align:super;">BETA</span></button>
        </div>

        <!-- ПОМОЩЬ -->
        <div id="help" class="tab-content active">
            <h2 data-i18n="quick_start">Быстрый старт</h2>
            <div class="info-card">
                <h3 data-i18n="three_steps">3 шага до первой нейросети</h3>
                <ol>
                    <li data-i18n="step1"><strong>Выберите датасет</strong> — Вкладка "Каталог" — выберите книгу или текст и скачайте одним кликом</li>
                    <li data-i18n="step2"><strong>Создайте модель</strong> — Вкладка "Создать" — выберите пресет и нажмите кнопку</li>
                    <li data-i18n="step3"><strong>Запустите обучение</strong> — Вкладка "Обучение" — выберите модель, нажмите "Начать"</li>
                </ol>
                <p style="margin-top:10px;color:#10b981;" data-i18n="after_training_tip">После обучения — генерируйте текст на вкладке "Генерация" и сравнивайте итерации на вкладке "Сравнение"</p>
            </div>
            <div class="info-card">
                <h3 data-i18n="whats_new">Что нового в v2</h3>
                <ul>
                    <li data-i18n="new_catalog"><strong>Каталог датасетов</strong> — 25+ книг и текстов + поиск по HuggingFace</li>
                    <li data-i18n="new_reinforce"><strong>REINFORCE</strong> — модель учится на своих ошибках (policy gradient)</li>
                    <li data-i18n="new_metrics"><strong>6 метрик качества</strong> — diversity, coherence, repetition, length, vocabulary, naturalness</li>
                    <li data-i18n="new_analytics"><strong>Аналитика</strong> — перплексия, скорость, ETA, бенчмарки</li>
                    <li data-i18n="new_compare"><strong>Сравнение</strong> — как модель улучшается на одних промптах</li>
                    <li data-i18n="new_confidence"><strong>Confidence</strong> — уверенность модели в каждом слове</li>
                    <li data-i18n="new_before_after"><strong>До/После</strong> — необученная vs обученная модель</li>
                </ul>
            </div>
            <div class="info-card">
                <h3 data-i18n="configurations">Конфигурации</h3>
                <table>
                    <tr><th data-i18n="th_goal">Цель</th><th>D Model</th><th>Layers</th><th>Iterations</th><th data-i18n="th_time">Время</th></tr>
                    <tr><td data-i18n="preset_quick">Быстрый тест</td><td>128</td><td>4</td><td>1,000</td><td data-i18n="time_5min">~5 мин</td></tr>
                    <tr><td data-i18n="preset_standard">Стандарт</td><td>256</td><td>6</td><td>10,000</td><td data-i18n="time_2h">~2 часа</td></tr>
                    <tr><td data-i18n="preset_quality">Качество</td><td>512</td><td>12</td><td>100,000</td><td data-i18n="time_24h">~24 часа</td></tr>
                </table>
            </div>
            <div class="info-card">
                <h3>FAQ</h3>
                <p data-i18n="faq_loss"><strong>Loss не падает?</strong> — Уменьшите learning rate или увеличьте модель.</p>
                <p data-i18n="faq_repeat"><strong>Повторяет один текст?</strong> — Overfitting. Больше данных или меньше итераций.</p>
                <p data-i18n="faq_memory"><strong>Out of memory?</strong> — Уменьшите batch_size или d_model.</p>
                <p data-i18n="faq_continue"><strong>Как продолжить?</strong> — Stop, потом заново — продолжит с чекпоинта.</p>
            </div>
        </div>

        <!-- КАТАЛОГ -->
        <div id="catalog" class="tab-content">
            <h2 data-i18n="catalog_title">Каталог датасетов</h2>
            <p style="color:#888;margin-bottom:15px;" data-i18n="catalog_desc">Скачайте готовые датасеты одним кликом или найдите на HuggingFace</p>
            <div class="category-filters" id="category_filters"></div>
            <div class="search-box">
                <input type="text" id="hf_search" placeholder="Поиск на HuggingFace (например: russian text, poetry, code)..." data-i18n-placeholder="hf_placeholder">
                <button class="btn btn-primary" onclick="searchHF()" data-i18n="btn_search_hf">Поиск HF</button>
            </div>
            <div class="search-box" style="margin-top:10px;">
                <input type="text" id="custom_url_name" placeholder="Название" style="flex:0.3;" data-i18n-placeholder="placeholder_name">
                <input type="text" id="custom_url" placeholder="URL (https://...)" style="flex:0.5;">
                <select id="custom_url_lang" style="flex:0.15;padding:10px;background:#16213e;color:#e0e0e0;border:1px solid #444;border-radius:8px;">
                    <option value="auto">Auto</option><option value="ru">RU</option><option value="en">EN</option>
                </select>
                <button class="btn btn-success" onclick="addCustomURL()" data-i18n="btn_add_url">+ URL</button>
            </div>
            <div id="catalog_download_status" style="display:none;" class="alert alert-info"></div>
            <div class="grid-3" id="catalog_grid"></div>
        </div>

        <!-- СОЗДАТЬ -->
        <div id="create" class="tab-content">
            <h2 data-i18n="btn_create_model">Создать модель</h2>
            <div class="info-card">
                <h3 data-i18n="choose_preset">Выберите пресет</h3>
                <div class="grid-3">
                    <div class="preset-card" onclick="applyPreset('quick')" id="preset_quick">
                        <h4 data-i18n="preset_quick">Быстрый тест</h4>
                        <p class="preset-desc" data-i18n="preset_quick_desc">~5 минут, базовое качество</p>
                        <p class="preset-specs">128d / 4 layers / 1K iter</p>
                    </div>
                    <div class="preset-card" onclick="applyPreset('standard')" id="preset_standard">
                        <h4 data-i18n="preset_standard">Стандарт</h4>
                        <p class="preset-desc" data-i18n="preset_standard_desc">~2 часа, хорошее качество</p>
                        <p class="preset-specs">256d / 6 layers / 10K iter</p>
                    </div>
                    <div class="preset-card" onclick="applyPreset('quality')" id="preset_quality">
                        <h4 data-i18n="preset_quality">Качество</h4>
                        <p class="preset-desc" data-i18n="preset_quality_desc">~24 часа, лучший результат</p>
                        <p class="preset-specs">512d / 12 layers / 100K iter</p>
                    </div>
                </div>
                <div id="hw_recommendation" style="margin-top:10px;"></div>
            </div>
            <div class="form-group">
                <label data-i18n="model_name">Название модели:</label>
                <input type="text" id="model_name" placeholder="my_model">
            </div>
            <div class="grid">
                <div class="form-group">
                    <label>Vocab Size: <span class="help-icon">?<span class="tooltip" data-i18n="tip_vocab">Размер словаря. 10000-25000 оптимально.</span></span></label>
                    <input type="number" id="vocab_size" value="8000">
                </div>
                <div class="form-group">
                    <label>D Model: <span class="help-icon">?<span class="tooltip" data-i18n="tip_dmodel">Размерность модели. Больше = умнее но медленнее.</span></span></label>
                    <input type="number" id="d_model" value="256">
                </div>
                <div class="form-group">
                    <label>Num Layers: <span class="help-icon">?<span class="tooltip" data-i18n="tip_layers">Количество трансформер-блоков. 4-12 оптимально.</span></span></label>
                    <input type="number" id="num_layers" value="6">
                </div>
                <div class="form-group">
                    <label>Num Heads: <span class="help-icon">?<span class="tooltip" data-i18n="tip_heads">Attention heads. Должно делить d_model нацело.</span></span></label>
                    <input type="number" id="num_heads" value="8">
                </div>
                <div class="form-group">
                    <label>D FF:</label>
                    <input type="number" id="d_ff" value="1024">
                </div>
                <div class="form-group">
                    <label>Max Seq Len:</label>
                    <input type="number" id="max_seq_len" value="256">
                </div>
            </div>
            <button class="btn btn-primary" onclick="createModel()" data-i18n="btn_create_model">Создать модель</button>
            <div id="create_status"></div>
        </div>

        <!-- ДАТАСЕТЫ -->
        <div id="datasets" class="tab-content">
            <h2 data-i18n="manage_datasets">Управление датасетами</h2>
            <div class="info-card">
                <h3 data-i18n="upload_files">Загрузить файлы</h3>
                <input type="file" id="book_file" accept=".txt,.csv,.json,.jsonl,.pdf,.jpg,.jpeg,.png" onchange="uploadBook()" style="display:none;">
                <div style="display:flex;align-items:center;gap:12px;margin:10px 0;">
                    <button class="btn btn-outline" onclick="document.getElementById('book_file').click()" data-i18n="btn_choose_file">Выбрать файл</button>
                    <span id="chosen_file_name" style="color:#888;font-size:0.9em;" data-i18n="no_file_chosen">Файл не выбран</span>
                </div>
                <div class="recommended" data-i18n="supported_formats">Поддерживаются: .txt, .csv, .json, .jsonl, .pdf, .jpg, .png</div>
                <div id="upload_status"></div>
            </div>
            <div class="info-card">
                <h3 data-i18n="manage">Управление</h3>
                <div class="form-group">
                    <label data-i18n="select_model">Выберите модель:</label>
                    <select id="dataset_model_name" onchange="loadModelDatasets()"></select>
                </div>
                <h4 data-i18n="attached_datasets">Прикреплённые к модели:</h4>
                <div id="attached_datasets"></div>
                <h4 style="margin-top:15px;" data-i18n="available_datasets">Доступные датасеты:</h4>
                <div id="available_datasets"></div>
            </div>
        </div>

        <!-- ОБУЧЕНИЕ -->
        <div id="train" class="tab-content">
            <h2 data-i18n="training_title">Обучение модели</h2>
            <div class="form-group">
                <label data-i18n="select_model">Выбрать модель:</label>
                <select id="train_model_name"></select>
            </div>
            <div class="grid">
                <div class="form-group">
                    <label>Max Iterations: <span class="help-icon">?<span class="tooltip" data-i18n="tip_iterations">1000=5мин, 10000=2ч, 100000=24ч</span></span></label>
                    <input type="number" id="max_iterations" value="10000">
                </div>
                <div class="form-group">
                    <label>Batch Size:</label>
                    <input type="number" id="batch_size" value="16">
                </div>
                <div class="form-group">
                    <label>Learning Rate:</label>
                    <input type="number" id="learning_rate" value="0.0003" step="0.0001">
                </div>
                <div class="form-group">
                    <label>Save Every:</label>
                    <input type="number" id="save_every" value="1000">
                </div>
            </div>
            <button class="btn btn-primary" onclick="startTraining()" data-i18n="btn_start_training">Начать обучение</button>
            <button class="btn btn-danger" onclick="stopTraining()" data-i18n="btn_stop_training">Остановить</button>

            <div class="milestones" id="milestones">
                <div class="milestone" data-pct="1"><div class="milestone-dot">1</div><span class="milestone-label" data-i18n="ms_start">Старт</span></div>
                <div class="milestone" data-pct="5"><div class="milestone-dot">5</div><span class="milestone-label" data-i18n="ms_warmup">Разогрев</span></div>
                <div class="milestone" data-pct="25"><div class="milestone-dot">25</div><span class="milestone-label" data-i18n="ms_quarter">Четверть</span></div>
                <div class="milestone" data-pct="50"><div class="milestone-dot">50</div><span class="milestone-label" data-i18n="ms_half">Половина</span></div>
                <div class="milestone" data-pct="75"><div class="milestone-dot">75</div><span class="milestone-label" data-i18n="ms_almost">Почти</span></div>
                <div class="milestone" data-pct="100"><div class="milestone-dot">!</div><span class="milestone-label" data-i18n="ms_done">Готово</span></div>
            </div>

            <div class="whats-happening" id="whats_happening">
                <h4 data-i18n="whats_happening">Что сейчас происходит</h4>
                <p id="whats_happening_text"></p>
            </div>

            <div class="chart-container"><canvas id="trainingChart"></canvas></div>

            <div class="status-box">
                <h3 data-i18n="status_training">Статус обучения</h3>
                <div class="status-item"><span class="status-label" data-i18n="status_label">Статус:</span><span class="status-value" id="is_training" data-i18n="status_idle">Не запущено</span></div>
                <div class="progress-bar"><div class="progress-fill" id="progress" style="width:0%">0%</div></div>
                <div class="status-item"><span class="status-label" data-i18n="iteration_label">Итерация:</span><span class="status-value" id="current_iteration">0 / 0</span></div>
                <div class="status-item"><span class="status-label" data-i18n="loss_label">Loss:</span><span class="status-value" id="current_loss">--</span></div>
                <div class="status-item"><span class="status-label" data-i18n="reward_label">Reward:</span><span class="status-value" id="current_reward">--</span></div>
                <div class="status-item"><span class="status-label" data-i18n="perplexity_label">Perplexity:</span><span class="status-value" id="current_perplexity">--</span></div>
                <div class="status-item"><span class="status-label" data-i18n="speed_label">Скорость:</span><span class="status-value" id="tokens_per_sec">--</span></div>
                <div class="status-item"><span class="status-label" data-i18n="remaining_label">Осталось:</span><span class="status-value" id="eta">--</span></div>
                <div class="status-item"><span class="status-label" data-i18n="memory_label">Память:</span><span class="status-value" id="memory_usage">--</span></div>
            </div>
            <div class="status-box" id="reward_components_box" style="display:none;">
                <h3 data-i18n="reward_components">Компоненты награды</h3>
                <div id="reward_components_list"></div>
            </div>
        </div>

        <!-- ГЕНЕРАЦИЯ -->
        <div id="generate" class="tab-content">
            <h2 data-i18n="generation_title">Генерация текста</h2>
            <div class="form-group">
                <label data-i18n="select_model">Модель:</label>
                <select id="gen_model_name"></select>
            </div>
            <div class="form-group">
                <label data-i18n="generate_prompt">Промпт:</label>
                <textarea id="prompt" rows="2" placeholder="Искусственный интеллект это..." data-i18n-placeholder="prompt_placeholder"></textarea>
            </div>
            <div class="grid">
                <div class="form-group">
                    <label>Max Length:</label>
                    <input type="number" id="max_length" value="100">
                </div>
                <div class="form-group">
                    <label>Temperature: <span class="help-icon">?<span class="tooltip" data-i18n="tip_temperature">0.5=консервативно, 1.0=норма, 1.5+=креативно</span></span></label>
                    <input type="number" id="temperature" value="0.8" step="0.1">
                </div>
                <div class="form-group">
                    <label>Top K:</label>
                    <input type="number" id="top_k" value="40">
                </div>
            </div>
            <button class="btn btn-primary" onclick="generateText()" data-i18n="btn_generate">Сгенерировать</button>
            <button class="btn btn-success" onclick="generateBeforeAfter()" data-i18n="before_after">До / После</button>
            <div class="info-card" style="margin-top:15px;">
                <h3 data-i18n="generation_result">Результат</h3> <span style="font-size:0.7em;color:#888;" data-i18n="confidence_hint">(цвет = уверенность модели: зелёный=высокая, жёлтый=средняя, красный=низкая)</span>
                <div id="generated_output" class="token-viz" style="min-height:80px;" data-i18n="text_placeholder">Текст появится здесь...</div>
            </div>
            <div id="quality_radar_container" style="display:none;" class="info-card">
                <h3 data-i18n="quality_metrics">Качество генерации</h3>
                <div class="radar-container"><canvas id="qualityRadar"></canvas></div>
            </div>
            <div id="before_after_container" style="display:none;">
                <h3 style="color:#667eea;margin:15px 0 10px;" data-i18n="before_vs_after">До обучения vs После</h3>
                <div class="compare-grid">
                    <div class="compare-panel"><h4 data-i18n="before_training">До обучения</h4><div id="before_text" class="text-output"></div><div id="before_quality" style="margin-top:8px;font-size:0.8em;color:#888;"></div></div>
                    <div class="compare-panel"><h4 data-i18n="after_training">После обучения</h4><div id="after_text" class="text-output"></div><div id="after_quality" style="margin-top:8px;font-size:0.8em;color:#888;"></div></div>
                </div>
                <div id="improvement_summary" class="alert alert-success" style="margin-top:10px;display:none;"></div>
            </div>
        </div>

        <!-- СРАВНЕНИЕ -->
        <div id="compare" class="tab-content">
            <h2 data-i18n="compare_checkpoints">Сравнение итераций</h2>
            <div class="form-group"><label data-i18n="select_model">Модель:</label><select id="compare_model_name" onchange="loadCheckpoints()"></select></div>
            <div class="grid">
                <div class="form-group"><label data-i18n="checkpoint_a_label">Чекпоинт A:</label><select id="checkpoint_a"></select></div>
                <div class="form-group"><label data-i18n="checkpoint_b_label">Чекпоинт B:</label><select id="checkpoint_b"></select></div>
            </div>
            <div class="form-group"><label data-i18n="compare_prompt_label">Промпт:</label><input type="text" id="compare_prompt" value="Искусственный интеллект" data-i18n-value="compare_default_prompt"></div>
            <button class="btn btn-primary" onclick="compareCheckpoints()" data-i18n="btn_compare">Сравнить</button>
            <div id="compare_results" style="display:none;margin-top:15px;">
                <div class="compare-grid">
                    <div class="compare-panel"><h4 id="compare_label_a">A</h4><div id="compare_text_a" class="text-output"></div><div id="compare_quality_a" style="margin-top:8px;font-size:0.8em;color:#888;"></div></div>
                    <div class="compare-panel"><h4 id="compare_label_b">B</h4><div id="compare_text_b" class="text-output"></div><div id="compare_quality_b" style="margin-top:8px;font-size:0.8em;color:#888;"></div></div>
                </div>
            </div>
            <div id="compare_radar_container" style="display:none;" class="info-card"><h3 data-i18n="compare_quality">Сравнение качества</h3><div class="radar-container"><canvas id="compareRadar"></canvas></div></div>
            <div class="info-card" style="margin-top:15px;"><h3 data-i18n="iteration_history">История итераций</h3><button class="btn btn-outline" onclick="loadAnalyticsReports()" data-i18n="btn_load_reports">Загрузить отчёты</button><div id="analytics_table" style="margin-top:10px;"></div></div>
        </div>

        <!-- МОДЕЛИ -->
        <div id="models" class="tab-content">
            <h2 data-i18n="my_models">Мои модели</h2>
            <button class="btn btn-success" onclick="loadModelsList()" data-i18n="btn_refresh">Обновить</button>
            <div id="models_list" style="margin-top:15px;"></div>
        </div>

        <!-- АВТОПИЛОТ -->
        <div id="autopilot" class="tab-content">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <h2 data-i18n="autopilot_title" style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">LLM Автопилот</h2>
                <span style="background:linear-gradient(135deg,#f59e0b,#ef4444);color:#fff;font-size:0.7em;font-weight:700;padding:3px 10px;border-radius:20px;letter-spacing:1px;text-transform:uppercase;">BETA</span>
            </div>
            <p data-i18n="autopilot_desc">Опишите цель — AI подберёт данные, создаст и обучит модель автоматически</p>
            <div style="background:#f59e0b15;border:1px solid #f59e0b44;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:0.85em;color:#f59e0b;">
                <span style="font-weight:700;">BETA</span> — <span data-i18n="autopilot_beta_notice">Эта функция в режиме бета-тестирования. Результаты могут быть нестабильными. Требуется API ключ от OpenAI, Anthropic или локальная LLM.</span>
            </div>

            <div class="info-card">
                <h3 data-i18n="autopilot_provider_title">Настройки LLM</h3>
                <div class="grid" style="grid-template-columns:1fr 1fr;">
                    <div class="form-group">
                        <label data-i18n="autopilot_provider">Провайдер:</label>
                        <select id="ap_provider" onchange="updateProviderFields()">
                            <option value="google_ai" data-i18n-option="autopilot_opt_google">Google AI Studio (Free / Бесплатно)</option>
                            <option value="groq" data-i18n-option="autopilot_opt_groq">Groq (Free / Бесплатно)</option>
                            <option value="openai">OpenAI (GPT-4o)</option>
                            <option value="anthropic">Anthropic (Claude)</option>
                            <option value="openai_compatible" data-i18n-option="autopilot_opt_local">Local / Ollama / LM Studio</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label data-i18n="autopilot_model">Модель:</label>
                        <input type="text" id="ap_model" placeholder="gemini-2.5-flash">
                    </div>
                </div>
                <div class="grid" style="grid-template-columns:1fr 1fr;">
                    <div class="form-group" id="ap_api_key_group">
                        <label data-i18n="autopilot_api_key">API ключ:</label>
                        <input type="password" id="ap_api_key" placeholder="sk-..." style="font-family:monospace;">
                    </div>
                    <div class="form-group" id="ap_endpoint_group">
                        <label data-i18n="autopilot_endpoint">Endpoint:</label>
                        <input type="text" id="ap_endpoint" placeholder="https://xxxx.ngrok.io/v1">
                    </div>
                </div>
                <div id="ap_provider_help" style="background:#667eea15;border:1px solid #667eea44;border-radius:8px;padding:12px 14px;margin-top:8px;font-size:0.82em;line-height:1.6;"></div>
            </div>

            <div class="info-card">
                <h3 data-i18n="autopilot_goal">Цель</h3>
                <div class="form-group">
                    <textarea id="ap_goal" rows="3" style="width:100%;resize:vertical;"
                        data-i18n-placeholder="autopilot_goal_placeholder"
                        placeholder="Хочу модель, которая пишет детективные рассказы в стиле Шерлока Холмса на английском"></textarea>
                </div>
                <div class="form-group" style="margin-top:8px;">
                    <label data-i18n="autopilot_time_budget">Лимит времени (минут):</label>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <input type="number" id="ap_time_budget" min="0" max="480" value="0" style="width:100px;">
                        <span style="color:#9ca3af;font-size:0.82em;" data-i18n="autopilot_time_hint">0 = без лимита. LLM подстроит кол-во итераций под время.</span>
                    </div>
                </div>
                <div style="display:flex;gap:10px;margin-top:10px;">
                    <button class="btn btn-primary" onclick="startAutopilot()" id="ap_start_btn" style="background:linear-gradient(135deg,#667eea,#764ba2);" data-i18n="autopilot_start">Запустить автопилот</button>
                    <button class="btn btn-danger" onclick="stopAutopilot()" id="ap_stop_btn" style="display:none;" data-i18n="autopilot_stop">Остановить</button>
                </div>
            </div>

            <div id="ap_state_box" class="alert alert-info" style="display:none;margin-top:15px;">
                <span id="ap_state_text"></span>
            </div>

            <div id="ap_log_box" class="status-box" style="display:none;margin-top:15px;">
                <h3 data-i18n="autopilot_log">Журнал действий</h3>
                <div id="ap_log_entries" style="max-height:500px;overflow-y:auto;font-size:0.85em;"></div>
            </div>
        </div>
    </div>

    <!-- Wizard Modal -->
    <div class="modal-overlay" id="wizard_modal">
        <div class="modal">
            <h2 data-i18n="wizard_welcome">Добро пожаловать!</h2>
            <p data-i18n="wizard_desc">Создайте свою первую нейросеть за 3 простых шага.</p>
            <div class="grid-3" style="margin:20px 0;">
                <div class="preset-card" onclick="wizardSelectPreset('quick')"><h4 data-i18n="preset_quick">Быстрый тест</h4><p class="preset-desc" data-i18n="time_5min">5 минут</p></div>
                <div class="preset-card" onclick="wizardSelectPreset('standard')"><h4 data-i18n="preset_standard">Стандарт</h4><p class="preset-desc" data-i18n="time_2h">2 часа</p></div>
                <div class="preset-card" onclick="wizardSelectPreset('quality')"><h4 data-i18n="preset_quality">Качество</h4><p class="preset-desc" data-i18n="time_24h">24 часа</p></div>
            </div>
            <button class="btn btn-outline" onclick="closeWizard()" data-i18n="btn_skip">Пропустить</button>
        </div>
    </div>

    <!-- Preview Modal -->
    <div class="modal-overlay" id="preview_modal">
        <div class="modal">
            <h2 id="preview_title" data-i18n="preview">Превью</h2>
            <pre id="preview_content" style="color:#aaa;font-size:0.85em;line-height:1.5;max-height:400px;overflow-y:auto;white-space:pre-wrap;"></pre>
            <button class="btn btn-outline" onclick="closePreview()" style="margin-top:15px;" data-i18n="btn_close">Закрыть</button>
        </div>
    </div>

<script>
let trainingChart = null;
let chartData = { labels: [], loss: [], reward: [], perplexity: [] };
let statusInterval = null;
let downloadInterval = null;
let currentCategory = null;

const PRESETS = {
    quick:    { d_model: 128, num_layers: 4, num_heads: 4, d_ff: 512,  max_seq_len: 128, vocab_size: 10000,  max_iterations: 1000,  batch_size: 8,  learning_rate: 0.001 },
    standard: { d_model: 256, num_layers: 6, num_heads: 8, d_ff: 1024, max_seq_len: 256, vocab_size: 15000,  max_iterations: 10000, batch_size: 16, learning_rate: 0.0003 },
    quality:  { d_model: 512, num_layers: 12, num_heads: 16, d_ff: 2048, max_seq_len: 512, vocab_size: 25000, max_iterations: 100000, batch_size: 32, learning_rate: 0.0001 },
};

var REWARD_LABELS = {
    diversity: 'Разнообразие', coherence: 'Когерентность', repetition_penalty: 'Без повторов',
    length_score: 'Длина', vocabulary_richness: 'Словарь', bigram_naturalness: 'Естественность'
};

function showTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById(tabName).classList.add('active');
    if (tabName === 'catalog') loadCatalog();
    if (tabName === 'create') loadHWRecommendation();
    if (tabName === 'datasets') { loadModelsSelect(); loadDatasets(); }
    if (tabName === 'train') { loadModelsSelect(); initChart(); startStatusUpdates(); }
    if (tabName === 'generate') loadModelsSelect();
    if (tabName === 'compare') { loadModelsSelect(); }
    if (tabName === 'models') loadModelsList();
}

// === CATALOG ===
async function loadCatalog(category) {
    currentCategory = category || null;
    let url = '/dataset_catalog';
    if (category) url += '?category=' + category;
    const catResp = await fetch('/dataset_catalog/categories');
    const catData = await catResp.json();
    const filtersDiv = document.getElementById('category_filters');
    filtersDiv.innerHTML = '<button class="cat-btn ' + (!category ? 'active' : '') + '" onclick="loadCatalog()">' + t('filter_all') + '</button>' +
        catData.categories.map(c => '<button class="cat-btn ' + (category===c.id ? 'active' : '') + '" onclick="loadCatalog(\'' + c.id + '\')">' + c.icon + ' ' + (currentLang === 'en' ? (c.name || c.name_ru) : c.name_ru) + ' (' + c.count + ')</button>').join('');
    const resp = await fetch(url);
    const data = await resp.json();
    const grid = document.getElementById('catalog_grid');
    grid.innerHTML = data.datasets.map(d => '<div class="catalog-card"><h4>' + (currentLang === 'en' ? (d.name || d.name_ru) : (d.name_ru || d.name)) + '</h4><p class="desc">' + (d.description || '') + '</p><div class="badges"><span class="badge badge-lang">' + d.language + '</span><span class="badge badge-size">' + d.size_estimate + '</span><span class="badge badge-cat">' + d.difficulty + '</span>' + (d.downloaded ? '<span class="badge badge-downloaded">' + t('already_downloaded') + '</span>' : '') + '</div><div style="margin-top:8px;">' + (d.downloaded ? '<button class="btn btn-outline" disabled>' + t('already_downloaded') + '</button>' : '<button class="btn btn-primary" onclick="downloadCatalogDS(\'' + d.id + '\')">' + t('btn_download_ds') + '</button>') + ' <button class="btn btn-outline" onclick="previewDS(\'' + d.id + '\')">' + t('btn_preview') + '</button></div></div>').join('');
}

async function downloadCatalogDS(id) {
    const statusDiv = document.getElementById('catalog_download_status');
    statusDiv.style.display = 'block';
    statusDiv.textContent = t('downloading');
    await fetch('/dataset_catalog/download/' + id, {method: 'POST'});
    downloadInterval = setInterval(async () => {
        const resp = await fetch('/dataset_catalog/download_status');
        const status = await resp.json();
        statusDiv.textContent = status.message || t('downloading');
        if (!status.is_downloading && status.progress !== 0) {
            clearInterval(downloadInterval);
            setTimeout(() => { statusDiv.style.display = 'none'; loadCatalog(currentCategory); }, 3000);
        }
    }, 1000);
}

async function previewDS(id) {
    document.getElementById('preview_content').textContent = t('loading');
    document.getElementById('preview_modal').classList.add('active');
    const resp = await fetch('/dataset_catalog/preview/' + id + '?lines=30');
    const data = await resp.json();
    document.getElementById('preview_title').textContent = data.name || t('preview');
    document.getElementById('preview_content').textContent = (data.lines || []).join('\n');
}
function closePreview() { document.getElementById('preview_modal').classList.remove('active'); }

async function searchHF() {
    const query = document.getElementById('hf_search').value;
    if (!query) return;
    const grid = document.getElementById('catalog_grid');
    grid.innerHTML = '<p style="color:#888;">' + t('searching') + '</p>';
    const resp = await fetch('/dataset_catalog/search_hf?query=' + encodeURIComponent(query) + '&limit=12');
    const data = await resp.json();
    if (data.results && data.results.length > 0 && !data.results[0].error) {
        grid.innerHTML = data.results.map(d => '<div class="catalog-card"><h4>' + d.name + '</h4><p class="desc">' + (d.description || 'HuggingFace dataset') + '</p><div class="badges"><span class="badge badge-cat">HuggingFace</span>' + (d.downloads ? '<span class="badge badge-size">' + d.downloads.toLocaleString() + ' downloads</span>' : '') + '</div></div>').join('');
    } else {
        grid.innerHTML = '<p style="color:#888;">' + t('nothing_found') + '</p>';
    }
}

async function addCustomURL() {
    var name = document.getElementById('custom_url_name').value;
    var url = document.getElementById('custom_url').value;
    var lang = document.getElementById('custom_url_lang').value;
    if (!name || !url) { alert(t('fill_name_url')); return; }
    var formData = new FormData();
    formData.append('name', name);
    formData.append('url', url);
    formData.append('language', lang);
    try {
        var resp = await fetch('/dataset_catalog/add_custom', { method: 'POST', body: formData });
        var data = await resp.json();
        if (data.status === 'added') {
            document.getElementById('custom_url_name').value = '';
            document.getElementById('custom_url').value = '';
            loadCatalog();
        }
    } catch(e) { alert('Error: ' + e.message); }
}

// === CREATE ===
function applyPreset(name) {
    const p = PRESETS[name];
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
    const el = document.getElementById('preset_' + name);
    if (el) el.classList.add('selected');
    Object.keys(p).forEach(key => { const e = document.getElementById(key); if (e) e.value = p[key]; });
}

async function loadHWRecommendation() {
    try {
        const resp = await fetch('/hardware_recommendation');
        const data = await resp.json();
        document.getElementById('hw_recommendation').innerHTML = '<div class="alert alert-info">' + t('hw_recommendation') + ': <strong>' + data.preset.toUpperCase() + '</strong> — ' + data.reason + '</div>';
    } catch(e) {}
}

async function createModel() {
    const config = {
        name: document.getElementById('model_name').value,
        vocab_size: parseInt(document.getElementById('vocab_size').value),
        d_model: parseInt(document.getElementById('d_model').value),
        num_layers: parseInt(document.getElementById('num_layers').value),
        num_heads: parseInt(document.getElementById('num_heads').value),
        d_ff: parseInt(document.getElementById('d_ff').value),
        max_seq_len: parseInt(document.getElementById('max_seq_len').value)
    };
    if (!config.name) { alert(t('enter_model_name')); return; }
    try {
        const resp = await fetch('/create_model', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(config) });
        const data = await resp.json();
        document.getElementById('create_status').innerHTML = data.status === 'success'
            ? '<div class="alert alert-success">' + t('model_created') + data.parameters.toLocaleString() + '</div>'
            : '<div class="alert alert-error">' + (data.detail || 'Error') + '</div>';
    } catch(e) { document.getElementById('create_status').innerHTML = '<div class="alert alert-error">' + e.message + '</div>'; }
}

// === DATASETS ===
async function uploadBook() {
    const fileInput = document.getElementById('book_file');
    if (!fileInput.files[0]) return;
    document.getElementById('chosen_file_name').textContent = fileInput.files[0].name;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
        const resp = await fetch('/upload_book', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.status === 'success') {
            document.getElementById('upload_status').innerHTML = '<div class="alert alert-success">' + t('uploaded') + ': ' + data.filename + ' (' + (data.size/1024).toFixed(1) + ' KB)</div>';
            loadDatasets();
        }
    } catch(e) { document.getElementById('upload_status').innerHTML = '<div class="alert alert-error">' + e.message + '</div>'; }
}

async function loadModelsSelect() {
    const resp = await fetch('/models');
    const data = await resp.json();
    ['train_model_name', 'gen_model_name', 'dataset_model_name', 'compare_model_name'].forEach(id => {
        const sel = document.getElementById(id);
        if (sel) sel.innerHTML = data.models.map(m => '<option value="' + m.name + '">' + m.name + '</option>').join('');
    });
}

async function loadDatasets() {
    const resp = await fetch('/books');
    const data = await resp.json();
    const div = document.getElementById('available_datasets');
    if (div) {
        div.innerHTML = data.books.map(b => '<div class="dataset-item"><span>' + (b.name||b) + ' (' + ((b.size||0)/1024).toFixed(1) + ' KB)</span><button class="btn btn-primary" onclick="attachDS(\'' + (b.name||b) + '\')">' + t('btn_attach') + '</button></div>').join('') || '<p style="color:#666;">' + t('no_datasets') + '</p>';
    }
    loadModelDatasets();
}

async function loadModelDatasets() {
    const modelName = document.getElementById('dataset_model_name') ? document.getElementById('dataset_model_name').value : '';
    if (!modelName) return;
    try {
        const resp = await fetch('/model_datasets/' + modelName);
        const data = await resp.json();
        const div = document.getElementById('attached_datasets');
        if (div) {
            div.innerHTML = data.attached.map(d => '<div class="dataset-item"><span>' + d.name + ' (' + (d.size/1024).toFixed(1) + ' KB)</span><button class="btn btn-danger" onclick="detachDS(\'' + d.name + '\')">' + t('btn_detach') + '</button></div>').join('') || '<p style="color:#666;">' + t('no_attached') + '</p>';
        }
    } catch(e) {}
}

async function attachDS(name) {
    const modelName = document.getElementById('dataset_model_name').value;
    await fetch('/attach_dataset', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model_name: modelName, dataset_name: name})});
    loadModelDatasets();
}

async function detachDS(name) {
    const modelName = document.getElementById('dataset_model_name').value;
    await fetch('/detach_dataset', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model_name: modelName, dataset_name: name})});
    loadModelDatasets();
}

// === TRAINING ===
function initChart() {
    if (trainingChart) return;
    const ctx = document.getElementById('trainingChart');
    if (!ctx) return;
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [
                { label: 'Loss', data: chartData.loss, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', tension: 0.4, yAxisID: 'y' },
                { label: 'Reward', data: chartData.reward, borderColor: '#667eea', backgroundColor: 'rgba(102,126,234,0.1)', tension: 0.4, yAxisID: 'y1' },
                { label: 'Perplexity', data: chartData.perplexity, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.4, yAxisID: 'y', borderDash: [5,5] }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y: { type: 'linear', display: true, position: 'left', grid: {color:'#222'}, ticks: {color:'#888'} },
                y1: { type: 'linear', display: true, position: 'right', grid: {drawOnChartArea:false}, ticks: {color:'#888'} }
            },
            plugins: { legend: { labels: {color:'#aaa'} } }
        }
    });
}

async function startTraining() {
    const config = {
        model_name: document.getElementById('train_model_name').value,
        max_iterations: parseInt(document.getElementById('max_iterations').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        learning_rate: parseFloat(document.getElementById('learning_rate').value),
        save_every: parseInt(document.getElementById('save_every').value)
    };
    try {
        const resp = await fetch('/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(config)});
        const data = await resp.json();
        if (data.status === 'success') { startStatusUpdates(); }
        else { alert(data.detail || 'Error'); }
    } catch(e) { alert(e.message); }
}

async function stopTraining() { await fetch('/stop_training', {method:'POST'}); }

function startStatusUpdates() {
    if (statusInterval) clearInterval(statusInterval);
    updateTrainingStatus();
    statusInterval = setInterval(updateTrainingStatus, 2000);
}

function formatETA(s) {
    if (s < 0) return '--';
    if (s < 60) return s + ' ' + t('eta_sec');
    if (s < 3600) return Math.floor(s/60) + ' ' + t('eta_min');
    return Math.floor(s/3600) + t('eta_h') + ' ' + Math.floor((s%3600)/60) + t('eta_m');
}

function updateMilestones(pct) {
    document.querySelectorAll('.milestone').forEach(m => {
        m.classList.toggle('reached', pct >= parseInt(m.dataset.pct));
    });
}

function updateWhatsHappening(s) {
    const el = document.getElementById('whats_happening');
    const txt = document.getElementById('whats_happening_text');
    if (!s.is_training) { el.style.display = 'none'; return; }
    el.style.display = 'block';
    const p = s.max_iterations > 0 ? s.current_iteration / s.max_iterations : 0;
    let msg = '';
    if (p < 0.03) msg = t('wh_start');
    else if (p < 0.15) msg = 'Loss: ' + s.current_loss.toFixed(2) + '. ' + t('wh_early');
    else if (p < 0.4) msg = t('wh_mid');
    else if (p < 0.7) msg = t('wh_late');
    else if (p < 0.95) msg = t('wh_final');
    else msg = t('wh_done');
    txt.textContent = msg;
}

async function updateTrainingStatus() {
    try {
        const resp = await fetch('/training_status');
        const s = await resp.json();
        document.getElementById('is_training').textContent = s.is_training ? t('training_active') : t('training_idle');
        document.getElementById('is_training').style.color = s.is_training ? '#10b981' : '#888';
        const pct = s.max_iterations > 0 ? (s.current_iteration / s.max_iterations * 100) : 0;
        document.getElementById('progress').style.width = pct.toFixed(1) + '%';
        document.getElementById('progress').textContent = pct.toFixed(1) + '%';
        document.getElementById('current_iteration').textContent = s.current_iteration + ' / ' + s.max_iterations;
        document.getElementById('current_loss').textContent = s.current_loss ? s.current_loss.toFixed(4) : '--';
        document.getElementById('current_reward').textContent = s.current_reward ? s.current_reward.toFixed(4) : '--';
        document.getElementById('current_perplexity').textContent = s.perplexity ? s.perplexity.toFixed(1) : '--';
        document.getElementById('tokens_per_sec').textContent = s.tokens_per_sec ? s.tokens_per_sec.toFixed(0) + ' tok/s' : '--';
        document.getElementById('eta').textContent = formatETA(s.eta_seconds || -1);
        document.getElementById('memory_usage').textContent = s.memory_mb ? s.memory_mb.toFixed(0) + ' MB' : '--';
        updateMilestones(pct);
        updateWhatsHappening(s);
        if (s.reward_components && Object.keys(s.reward_components).length > 0) {
            document.getElementById('reward_components_box').style.display = 'block';
            document.getElementById('reward_components_list').innerHTML = Object.entries(s.reward_components).map(function(e) {
                return '<div class="status-item"><span class="status-label">' + (REWARD_LABELS[e[0]] || e[0]) + '</span><span class="status-value">' + e[1].toFixed(4) + '</span></div>';
            }).join('');
        }
        if (s.current_iteration > 0 && s.is_training) {
            chartData.labels.push(s.current_iteration);
            chartData.loss.push(s.current_loss);
            chartData.reward.push(s.current_reward);
            chartData.perplexity.push(s.perplexity || null);
            if (chartData.labels.length > 100) { chartData.labels.shift(); chartData.loss.shift(); chartData.reward.shift(); chartData.perplexity.shift(); }
            if (trainingChart) trainingChart.update();
        }
    } catch(e) {}
}

// === GENERATION ===
async function generateText() {
    const config = {
        model_name: document.getElementById('gen_model_name').value,
        prompt: document.getElementById('prompt').value,
        max_length: parseInt(document.getElementById('max_length').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_k: parseInt(document.getElementById('top_k').value)
    };
    document.getElementById('generated_output').innerHTML = '<span style="color:#888;">' + t('generating') + '</span>';
    try {
        const resp = await fetch('/generate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(config)});
        const data = await resp.json();
        if (data.token_details && data.token_details.length > 0) {
            document.getElementById('generated_output').innerHTML = data.token_details.map(function(t) {
                var hue = Math.round(t.confidence * 120);
                var bg = 'hsl(' + hue + ', 60%, 15%)';
                var border = 'hsl(' + hue + ', 60%, 35%)';
                var alts = t.top_alternatives ? t.top_alternatives.map(function(a) { return a.token + ' ' + (a.prob*100).toFixed(1) + '%'; }).join(', ') : '';
                return '<span class="token" style="background:' + bg + ';border:1px solid ' + border + ';" data-info="Confidence: ' + (t.confidence*100).toFixed(1) + '% | Alt: ' + alts + '">' + t.token + '</span>';
            }).join(' ');
        } else {
            document.getElementById('generated_output').textContent = data.generated_text || 'Error';
        }
        if (data.quality_metrics && data.quality_metrics.components) {
            renderRadar('qualityRadar', [data.quality_metrics], [t('radar_generation')]);
            document.getElementById('quality_radar_container').style.display = 'block';
        }
    } catch(e) { document.getElementById('generated_output').textContent = 'Error: ' + e.message; }
}

async function generateBeforeAfter() {
    const config = {
        model_name: document.getElementById('gen_model_name').value,
        prompt: document.getElementById('prompt').value,
        max_length: parseInt(document.getElementById('max_length').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_k: parseInt(document.getElementById('top_k').value)
    };
    document.getElementById('before_after_container').style.display = 'block';
    document.getElementById('before_text').textContent = t('loading');
    document.getElementById('after_text').textContent = t('loading');
    try {
        const resp = await fetch('/generate_before_after', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(config)});
        const data = await resp.json();
        document.getElementById('before_text').textContent = data.before ? data.before.text : 'N/A';
        document.getElementById('after_text').textContent = data.after ? data.after.text : 'N/A';
        document.getElementById('before_quality').textContent = 'Reward: ' + (data.before && data.before.quality ? data.before.quality.total.toFixed(4) : '0');
        document.getElementById('after_quality').textContent = 'Reward: ' + (data.after && data.after.quality ? data.after.quality.total.toFixed(4) : '0');
        if (data.improvement) {
            var summary = document.getElementById('improvement_summary');
            var delta = data.improvement.reward_delta;
            summary.style.display = 'block';
            summary.className = delta > 0 ? 'alert alert-success' : 'alert alert-info';
            var details = Object.entries(data.improvement.components_delta || {}).filter(function(e) { return Math.abs(e[1]) > 0.01; }).map(function(e) { return (REWARD_LABELS[e[0]] || e[0]) + ': ' + (e[1] > 0 ? '+' : '') + e[1].toFixed(3); }).join(', ');
            summary.textContent = delta > 0 ? t('improvement') + delta.toFixed(4) + '! ' + details : 'Reward: ' + delta.toFixed(4) + '. ' + details;
        }
    } catch(e) { document.getElementById('before_text').textContent = 'Error: ' + e.message; }
}

// === COMPARISON ===
async function loadCheckpoints() {
    var modelName = document.getElementById('compare_model_name').value;
    if (!modelName) return;
    var resp = await fetch('/checkpoints/' + modelName);
    var data = await resp.json();
    ['checkpoint_a', 'checkpoint_b'].forEach(function(id) {
        var sel = document.getElementById(id);
        sel.innerHTML = data.checkpoints.map(function(c) { return '<option value="' + c.iteration + '">Iteration ' + c.iteration + ' (' + c.size_mb + ' MB)</option>'; }).join('');
    });
}

async function compareCheckpoints() {
    var model = document.getElementById('compare_model_name').value;
    var iterA = parseInt(document.getElementById('checkpoint_a').value);
    var iterB = parseInt(document.getElementById('checkpoint_b').value);
    var prompt = document.getElementById('compare_prompt').value;
    if (!model || !prompt) return;
    document.getElementById('compare_results').style.display = 'block';
    document.getElementById('compare_text_a').textContent = t('generating');
    document.getElementById('compare_text_b').textContent = t('generating');
    try {
        var resp = await fetch('/compare_generations', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model_name: model, prompt: prompt, iterations: [iterA, iterB], max_length: 100})});
        var data = await resp.json();
        if (data.comparisons && data.comparisons.length >= 2) {
            var a = data.comparisons[0], b = data.comparisons[1];
            document.getElementById('compare_label_a').textContent = 'Iteration ' + a.iteration;
            document.getElementById('compare_label_b').textContent = 'Iteration ' + b.iteration;
            document.getElementById('compare_text_a').textContent = a.text || a.error || 'N/A';
            document.getElementById('compare_text_b').textContent = b.text || b.error || 'N/A';
            document.getElementById('compare_quality_a').textContent = a.quality ? 'Reward: ' + a.quality.total.toFixed(4) : '';
            document.getElementById('compare_quality_b').textContent = b.quality ? 'Reward: ' + b.quality.total.toFixed(4) : '';
            if (a.quality && b.quality) {
                renderRadar('compareRadar', [a.quality, b.quality], ['Iter ' + a.iteration, 'Iter ' + b.iteration]);
                document.getElementById('compare_radar_container').style.display = 'block';
            }
        }
    } catch(e) { document.getElementById('compare_text_a').textContent = 'Error: ' + e.message; }
}

async function loadAnalyticsReports() {
    try {
        var resp = await fetch('/training_analytics/reports');
        var data = await resp.json();
        var div = document.getElementById('analytics_table');
        if (data.reports && data.reports.length > 0) {
            div.innerHTML = '<table><tr><th>Iter</th><th>Loss</th><th>Perplexity</th><th>Reward</th><th>Speed</th></tr>' +
                data.reports.map(function(r) { return '<tr><td>' + r.iteration + '</td><td>' + r.loss.toFixed(4) + '</td><td>' + r.perplexity.toFixed(1) + '</td><td>' + (r.reward && r.reward.total !== undefined ? r.reward.total.toFixed(4) : '--') + '</td><td>' + (r.tokens_per_sec ? r.tokens_per_sec.toFixed(0) + ' tok/s' : '--') + '</td></tr>'; }).join('') + '</table>';
        } else { div.innerHTML = '<p style="color:#888;">' + t('no_data') + '</p>'; }
    } catch(e) {}
}

// === RADAR ===
function renderRadar(canvasId, qualities, labels) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    var colors = ['#667eea', '#10b981', '#f59e0b', '#ef4444'];
    var radarLabels = Object.keys(REWARD_LABELS).map(function(k) { return REWARD_LABELS[k]; });
    new Chart(canvas, {
        type: 'radar',
        data: {
            labels: radarLabels,
            datasets: qualities.map(function(q, i) { return {
                label: labels[i] || '#' + (i+1),
                data: Object.keys(REWARD_LABELS).map(function(k) { return q.components ? (q.components[k] || 0) : 0; }),
                borderColor: colors[i % colors.length],
                backgroundColor: colors[i % colors.length] + '22',
                pointBackgroundColor: colors[i % colors.length]
            }; })
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { r: { min: 0, max: 1, grid: {color:'#333'}, angleLines: {color:'#333'}, pointLabels: {color:'#aaa', font:{size:10}}, ticks: {display:false} } },
            plugins: { legend: { labels: {color:'#aaa'} } }
        }
    });
}

// === MODELS ===
async function loadModelsList() {
    var resp = await fetch('/models');
    var data = await resp.json();
    var div = document.getElementById('models_list');
    div.innerHTML = data.models.map(function(m) {
        return '<div class="info-card"><h3>' + m.name + (m.trained ? ' <span style="color:#10b981;">(' + t('trained') + ')</span>' : '') + '</h3>'
            + '<p>d_model=' + m.config.d_model + ', layers=' + m.config.num_layers + ', heads=' + m.config.num_heads + ', vocab=' + m.config.vocab_size + '</p>'
            + '<p>' + t('datasets_count') + ': ' + (m.total_datasets || 0) + ' | ' + t('size') + ': ' + m.size_mb + ' MB</p>'
            + '<button class="btn btn-success" onclick="window.location.href=\'/download_model/' + m.name + '\'">' + t('btn_download') + '</button>'
            + '<button class="btn btn-danger" onclick="deleteModel(\'' + m.name + '\')">' + t('btn_delete') + '</button>'
            + '</div>';
    }).join('') || '<p style="color:#888;">' + t('no_models') + '</p>';
}

async function deleteModel(name) {
    if (!confirm(t('confirm_delete') + ' "' + name + '"?')) return;
    try {
        var resp = await fetch('/models/' + name, { method: 'DELETE' });
        var data = await resp.json();
        if (data.status === 'deleted') {
            loadModelsList();
            loadModelsSelect();
        }
    } catch(e) { alert('Error: ' + e.message); }
}

// === WIZARD ===
async function checkWizard() {
    try {
        var resp = await fetch('/models');
        var data = await resp.json();
        if (data.models.length === 0) document.getElementById('wizard_modal').classList.add('active');
    } catch(e) {}
}
function wizardSelectPreset(name) {
    closeWizard();
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.tab-content').forEach(function(c) { c.classList.remove('active'); });
    document.querySelectorAll('.tab')[2].classList.add('active');
    document.getElementById('create').classList.add('active');
    applyPreset(name);
}
function closeWizard() { document.getElementById('wizard_modal').classList.remove('active'); }

// === LANGUAGE TOGGLE ===
var currentLang = localStorage.getItem('azr_lang') || 'ru';
var TRANSLATIONS = {
    ru: {
        subtitle: 'Создавайте, обучайте и анализируйте нейросети — просто и наглядно',
        tab_help: 'Помощь', tab_catalog: 'Каталог', tab_create: 'Создать',
        tab_datasets: 'Датасеты', tab_train: 'Обучение', tab_generate: 'Генерация',
        tab_compare: 'Сравнение', tab_models: 'Модели',
        quick_start: 'Быстрый старт',
        three_steps: '3 шага до первой нейросети',
        step1: 'Выберите датасет — Вкладка "Каталог" — выберите книгу или текст и скачайте одним кликом',
        step2: 'Создайте модель — Вкладка "Создать" — выберите пресет и нажмите кнопку',
        step3: 'Запустите обучение — Вкладка "Обучение" — выберите модель, нажмите "Начать"',
        after_training_tip: 'После обучения — генерируйте текст на вкладке "Генерация" и сравнивайте итерации на вкладке "Сравнение"',
        whats_new: 'Что нового в v2',
        new_catalog: 'Каталог датасетов — 25+ книг и текстов + поиск по HuggingFace',
        new_reinforce: 'REINFORCE — модель учится на своих ошибках (policy gradient)',
        new_metrics: '6 метрик качества — diversity, coherence, repetition, length, vocabulary, naturalness',
        new_analytics: 'Аналитика — перплексия, скорость, ETA, бенчмарки',
        new_compare: 'Сравнение — как модель улучшается на одних промптах',
        new_confidence: 'Confidence — уверенность модели в каждом слове',
        new_before_after: 'До/После — необученная vs обученная модель',
        configurations: 'Конфигурации',
        th_goal: 'Цель', th_time: 'Время',
        time_5min: '~5 мин', time_2h: '~2 часа', time_24h: '~24 часа',
        faq_loss: 'Loss не падает? — Уменьшите learning rate или увеличьте модель.',
        faq_repeat: 'Повторяет один текст? — Overfitting. Больше данных или меньше итераций.',
        faq_memory: 'Out of memory? — Уменьшите batch_size или d_model.',
        faq_continue: 'Как продолжить? — Stop, потом заново — продолжит с чекпоинта.',
        catalog_title: 'Каталог датасетов',
        catalog_desc: 'Скачайте готовые датасеты одним кликом или найдите на HuggingFace',
        btn_search_hf: 'Поиск HF',
        btn_add_url: '+ URL',
        hf_placeholder: 'Поиск на HuggingFace (например: russian text, poetry, code)...',
        prompt_placeholder: 'Искусственный интеллект это...',
        choose_preset: 'Выберите пресет',
        preset_quick: 'Быстрый тест', preset_standard: 'Стандарт', preset_quality: 'Качество',
        preset_quick_desc: '~5 минут, базовое качество',
        preset_standard_desc: '~2 часа, хорошее качество',
        preset_quality_desc: '~24 часа, лучший результат',
        model_name: 'Название модели:',
        tip_vocab: 'Размер словаря. 10000-25000 оптимально.',
        tip_dmodel: 'Размерность модели. Больше = умнее но медленнее.',
        tip_layers: 'Количество трансформер-блоков. 4-12 оптимально.',
        tip_heads: 'Attention heads. Должно делить d_model нацело.',
        tip_iterations: '1000=5мин, 10000=2ч, 100000=24ч',
        tip_temperature: '0.5=консервативно, 1.0=норма, 1.5+=креативно',
        btn_create_model: 'Создать модель',
        manage_datasets: 'Управление датасетами',
        upload_files: 'Загрузить файлы',
        supported_formats: 'Поддерживаются: .txt, .csv, .json, .jsonl, .pdf, .jpg, .png',
        manage: 'Управление',
        select_model: 'Выберите модель:',
        attached_datasets: 'Прикреплённые к модели:',
        available_datasets: 'Доступные датасеты:',
        training_title: 'Обучение модели',
        btn_start_training: 'Начать обучение',
        btn_stop_training: 'Остановить',
        ms_start: 'Старт', ms_warmup: 'Разогрев', ms_quarter: 'Четверть',
        ms_half: 'Половина', ms_almost: 'Почти', ms_done: 'Готово',
        whats_happening: 'Что сейчас происходит',
        status_training: 'Статус обучения',
        status_label: 'Статус:', iteration_label: 'Итерация:',
        loss_label: 'Loss:', reward_label: 'Reward:', perplexity_label: 'Perplexity:',
        speed_label: 'Скорость:', remaining_label: 'Осталось:', memory_label: 'Память:',
        training_idle: 'Не активно', training_active: 'Обучается...',
        reward_components: 'Компоненты награды',
        generation_title: 'Генерация текста',
        generate_prompt: 'Промпт:',
        btn_generate: 'Сгенерировать',
        before_after: 'До / После',
        generation_result: 'Результат',
        confidence_hint: '(цвет = уверенность модели: зелёный=высокая, жёлтый=средняя, красный=низкая)',
        text_placeholder: 'Текст появится здесь...',
        quality_metrics: 'Качество генерации',
        before_vs_after: 'До обучения vs После',
        before_training: 'До обучения', after_training: 'После обучения',
        compare_checkpoints: 'Сравнение итераций',
        btn_compare: 'Сравнить',
        compare_quality: 'Сравнение качества',
        iteration_history: 'История итераций',
        btn_load_reports: 'Загрузить отчёты',
        my_models: 'Мои модели',
        btn_refresh: 'Обновить',
        btn_download: 'Скачать',
        btn_delete: 'Удалить',
        trained: 'Обучена',
        datasets_count: 'Датасетов',
        size: 'Размер',
        confirm_delete: 'Удалить модель',
        no_models: 'Нет моделей.',
        fill_name_url: 'Введите название и URL',
        wizard_welcome: 'Добро пожаловать!',
        wizard_desc: 'Создайте свою первую нейросеть за 3 простых шага.',
        btn_skip: 'Пропустить',
        preview: 'Превью',
        btn_close: 'Закрыть',
        hw_recommendation: 'Рекомендация',
        status_idle: 'Не запущено',
        checkpoint_a_label: 'Чекпоинт A:',
        checkpoint_b_label: 'Чекпоинт B:',
        compare_prompt_label: 'Промпт:',
        compare_default_prompt: 'Искусственный интеллект',
        placeholder_name: 'Название',
        uploaded: 'Загружено',
        btn_choose_file: 'Выбрать файл',
        no_file_chosen: 'Файл не выбран',
        radar_generation: 'Генерация',
        eta_sec: 'сек', eta_min: 'мин', eta_h: 'ч', eta_m: 'м',
        tab_autopilot: 'Автопилот',
        autopilot_title: 'LLM Автопилот',
        autopilot_desc: 'Опишите цель — AI подберёт данные, создаст и обучит модель автоматически',
        autopilot_provider_title: 'Настройки LLM',
        autopilot_provider: 'Провайдер:',
        autopilot_model: 'Модель:',
        autopilot_model_hint: 'gpt-4o',
        autopilot_api_key: 'API ключ:',
        autopilot_endpoint: 'Endpoint:',
        autopilot_goal: 'Цель',
        autopilot_goal_placeholder: 'Хочу модель, которая пишет детективные рассказы в стиле Шерлока Холмса на английском',
        autopilot_start: 'Запустить автопилот',
        autopilot_stop: 'Остановить',
        autopilot_log: 'Журнал действий',
        autopilot_enter_goal: 'Введите цель для автопилота',
        autopilot_enter_key: 'Введите API ключ',
        autopilot_beta_notice: 'Эта функция в режиме бета-тестирования. Результаты могут быть нестабильными. Требуется API ключ от OpenAI, Anthropic или локальная LLM.',
        autopilot_state_idle: 'Не запущен',
        autopilot_state_planning: 'Планирование...',
        autopilot_state_executing: 'Выполнение действий...',
        autopilot_state_monitoring: 'Мониторинг обучения...',
        autopilot_state_completed: 'Завершено!',
        autopilot_state_error: 'Ошибка',
        autopilot_state_stopped: 'Остановлен',
        autopilot_time_budget: 'Лимит времени (минут):',
        autopilot_time_hint: '0 = без лимита. LLM подстроит кол-во итераций под время.',
        autopilot_opt_google: 'Google AI Studio (Бесплатно)',
        autopilot_opt_groq: 'Groq (Бесплатно)',
        autopilot_opt_local: 'Локальная / Ollama / LM Studio',
        autopilot_help_google_ai: '<b>Google AI Studio — бесплатный Gemini (рекомендуем):</b><br><br>1. Откройте <a href="https://aistudio.google.com/apikey" target="_blank" style="color:#667eea;text-decoration:underline;">aistudio.google.com/apikey</a> (войдите через Google)<br>2. Нажмите <b>«Create API Key»</b><br>3. Скопируйте ключ<br>4. Вставьте его в поле <b>API ключ</b> выше<br><br>💡 <b>Полностью бесплатно</b> — без карты, без оплаты. Gemini 2.5 Flash — мощная модель с огромным лимитом (1 млн токенов/мин). Идеально подходит для tool calling.',
        autopilot_help_groq: '<b>Groq — бесплатная и быстрая LLM:</b><br><br>1. Откройте <a href="https://console.groq.com/login" target="_blank" style="color:#667eea;text-decoration:underline;">console.groq.com</a> и войдите через Google (1 клик)<br>2. Перейдите в <a href="https://console.groq.com/keys" target="_blank" style="color:#667eea;text-decoration:underline;">API Keys</a><br>3. Нажмите <b>«Create API Key»</b><br>4. Скопируйте ключ (начинается с <code>gsk_...</code>)<br>5. Вставьте его в поле <b>API ключ</b> выше<br><br>⚠️ Бесплатно, но лимит 12K токенов/мин — могут быть паузы. Если часто ошибка rate limit — используйте Google AI Studio.',
        autopilot_help_openai: '<b>OpenAI (платный):</b><br><br><b>Где получить API ключ:</b><br>1. Откройте <a href="https://platform.openai.com/signup" target="_blank" style="color:#667eea;text-decoration:underline;">platform.openai.com/signup</a> и создайте аккаунт (или войдите через Google)<br>2. Откройте <a href="https://platform.openai.com/api-keys" target="_blank" style="color:#667eea;text-decoration:underline;">platform.openai.com/api-keys</a><br>3. Нажмите кнопку <b>«+ Create new secret key»</b><br>4. Скопируйте ключ (начинается с <code>sk-...</code>) — он показывается только один раз!<br>5. Вставьте его в поле <b>API ключ</b> выше<br><br><b>⚠️ Нужно пополнить баланс:</b> перейдите в <a href="https://platform.openai.com/account/billing" target="_blank" style="color:#667eea;text-decoration:underline;">Billing</a> → Add payment method → пополните от $5<br><br>💰 Стоимость: ~$0.01–0.05 за один запуск автопилота (GPT-4o). $5 хватит на сотни запусков.',
        autopilot_help_anthropic: '<b>Anthropic Claude (платный):</b><br><br><b>Где получить API ключ:</b><br>1. Откройте <a href="https://console.anthropic.com/login" target="_blank" style="color:#667eea;text-decoration:underline;">console.anthropic.com</a> и создайте аккаунт<br>2. Откройте <a href="https://console.anthropic.com/settings/keys" target="_blank" style="color:#667eea;text-decoration:underline;">Settings → API Keys</a><br>3. Нажмите <b>«Create Key»</b><br>4. Скопируйте ключ (начинается с <code>sk-ant-...</code>) — он показывается только один раз!<br>5. Вставьте его в поле <b>API ключ</b> выше<br><br><b>⚠️ Нужно пополнить баланс:</b> перейдите в <a href="https://console.anthropic.com/settings/billing" target="_blank" style="color:#667eea;text-decoration:underline;">Billing</a> → Add payment method → пополните от $5<br><br>💰 Стоимость: ~$0.01–0.05 за один запуск автопилота. $5 хватит на сотни запусков.',
        autopilot_help_openai_compatible: '<b>Локальная LLM (бесплатно, но грузит ваш ПК):</b><br><br>🔑 <b>API ключ не нужен</b> — всё работает локально на вашем компьютере.<br><br><b>Вариант 1 — Ollama (проще):</b><br>1. Скачайте и установите с <a href="https://ollama.com/download" target="_blank" style="color:#667eea;text-decoration:underline;">ollama.com/download</a> (Windows/Mac/Linux)<br>2. Откройте терминал (командную строку) и выполните:<br>&nbsp;&nbsp;&nbsp;<code>ollama pull llama3</code> — скачает модель (~4 ГБ)<br>3. Ollama запустится автоматически после установки<br>4. Endpoint: <code>http://localhost:11434/v1</code> (уже заполнен)<br>5. В поле Модель напишите: <code>llama3</code><br><br><b>Вариант 2 — LM Studio (с интерфейсом):</b><br>1. Скачайте с <a href="https://lmstudio.ai" target="_blank" style="color:#667eea;text-decoration:underline;">lmstudio.ai</a> и установите<br>2. В программе найдите и скачайте модель (например llama3)<br>3. Перейдите во вкладку <b>«Local Server»</b> (иконка ↔ слева) → нажмите <b>«Start Server»</b><br>4. Endpoint: <code>http://localhost:1234/v1</code><br>5. В поле Модель укажите название загруженной модели<br><br>⚠️ <b>Требует мощный ПК</b> (16GB+ RAM, лучше с видеокартой). LLM работает одновременно с обучением — может быть медленно. Рекомендуем Google Colab если ПК слабый.',
        btn_attach: '+ Прикрепить', btn_detach: 'Открепить',
        filter_all: 'Все',
        no_datasets: 'Нет датасетов.', no_attached: 'Нет прикреплённых.',
        downloading: 'Скачивание...',
        already_downloaded: 'Уже скачан',
        btn_download_ds: 'Скачать', btn_preview: 'Превью',
        searching: 'Поиск...', nothing_found: 'Ничего не найдено.',
        no_data: 'Нет данных.',
        model_created: 'Модель создана! Параметров: ',
        enter_model_name: 'Введите название модели',
        generating: 'Генерация...',
        loading: 'Загрузка...',
        improvement: 'Улучшение reward на ',
        wh_start: 'Модель только начинает. Loss высокий — это нормально. Идёт запоминание базовых паттернов.',
        wh_early: 'Модель изучает частые слова и простые сочетания.',
        wh_mid: 'Базовые паттерны выучены. Reward растёт — качество генерации улучшается.',
        wh_late: 'Обучение идёт. Модель улавливает тонкие закономерности. Loss падает медленнее — это нормально.',
        wh_final: 'Финальная стадия! Последние корректировки. Скоро обучение завершится.',
        wh_done: 'Обучение почти завершено!',
        rl_diversity: 'Разнообразие', rl_coherence: 'Когерентность', rl_repetition: 'Без повторов',
        rl_length: 'Длина', rl_vocabulary: 'Словарь', rl_naturalness: 'Естественность',
    },
    en: {
        subtitle: 'Create, train and analyze neural networks — simple and visual',
        tab_help: 'Help', tab_catalog: 'Catalog', tab_create: 'Create',
        tab_datasets: 'Datasets', tab_train: 'Training', tab_generate: 'Generate',
        tab_compare: 'Compare', tab_models: 'Models',
        quick_start: 'Quick Start',
        three_steps: '3 steps to your first neural network',
        step1: 'Choose a dataset — "Catalog" tab — pick a book or text and download with one click',
        step2: 'Create a model — "Create" tab — choose a preset and click the button',
        step3: 'Start training — "Training" tab — select a model, press "Start"',
        after_training_tip: 'After training — generate text on the "Generate" tab and compare iterations on the "Compare" tab',
        whats_new: 'What\'s new in v2',
        new_catalog: 'Dataset Catalog — 25+ books and texts + HuggingFace search',
        new_reinforce: 'REINFORCE — model learns from its mistakes (policy gradient)',
        new_metrics: '6 quality metrics — diversity, coherence, repetition, length, vocabulary, naturalness',
        new_analytics: 'Analytics — perplexity, speed, ETA, benchmarks',
        new_compare: 'Comparison — how the model improves on the same prompts',
        new_confidence: 'Confidence — model certainty for each word',
        new_before_after: 'Before/After — untrained vs trained model',
        configurations: 'Configurations',
        th_goal: 'Goal', th_time: 'Time',
        time_5min: '~5 min', time_2h: '~2 hours', time_24h: '~24 hours',
        faq_loss: 'Loss not decreasing? — Reduce learning rate or increase model size.',
        faq_repeat: 'Repeating same text? — Overfitting. More data or fewer iterations.',
        faq_memory: 'Out of memory? — Reduce batch_size or d_model.',
        faq_continue: 'How to continue? — Stop, then restart — it will resume from checkpoint.',
        catalog_title: 'Dataset Catalog',
        catalog_desc: 'Download ready-made datasets with one click or find them on HuggingFace',
        btn_search_hf: 'Search HF',
        btn_add_url: '+ URL',
        hf_placeholder: 'Search HuggingFace (e.g.: russian text, poetry, code)...',
        prompt_placeholder: 'Artificial intelligence is...',
        choose_preset: 'Choose a preset',
        preset_quick: 'Quick Test', preset_standard: 'Standard', preset_quality: 'Quality',
        preset_quick_desc: '~5 minutes, basic quality',
        preset_standard_desc: '~2 hours, good quality',
        preset_quality_desc: '~24 hours, best results',
        model_name: 'Model name:',
        tip_vocab: 'Vocabulary size. 10000-25000 is optimal.',
        tip_dmodel: 'Model dimension. Larger = smarter but slower.',
        tip_layers: 'Number of transformer blocks. 4-12 is optimal.',
        tip_heads: 'Attention heads. Must evenly divide d_model.',
        tip_iterations: '1000=5min, 10000=2h, 100000=24h',
        tip_temperature: '0.5=conservative, 1.0=normal, 1.5+=creative',
        btn_create_model: 'Create Model',
        manage_datasets: 'Manage Datasets',
        upload_files: 'Upload Files',
        supported_formats: 'Supported: .txt, .csv, .json, .jsonl, .pdf, .jpg, .png',
        manage: 'Management',
        select_model: 'Select model:',
        attached_datasets: 'Attached to model:',
        available_datasets: 'Available datasets:',
        training_title: 'Model Training',
        btn_start_training: 'Start Training',
        btn_stop_training: 'Stop',
        ms_start: 'Start', ms_warmup: 'Warmup', ms_quarter: 'Quarter',
        ms_half: 'Half', ms_almost: 'Almost', ms_done: 'Done',
        whats_happening: 'What\'s happening now',
        status_training: 'Training Status',
        status_label: 'Status:', iteration_label: 'Iteration:',
        loss_label: 'Loss:', reward_label: 'Reward:', perplexity_label: 'Perplexity:',
        speed_label: 'Speed:', remaining_label: 'Remaining:', memory_label: 'Memory:',
        training_idle: 'Idle', training_active: 'Training...',
        reward_components: 'Reward Components',
        generation_title: 'Text Generation',
        generate_prompt: 'Prompt:',
        btn_generate: 'Generate',
        before_after: 'Before / After',
        generation_result: 'Result',
        confidence_hint: '(color = model confidence: green=high, yellow=medium, red=low)',
        text_placeholder: 'Text will appear here...',
        quality_metrics: 'Generation Quality',
        before_vs_after: 'Before Training vs After',
        before_training: 'Before Training', after_training: 'After Training',
        compare_checkpoints: 'Compare Iterations',
        btn_compare: 'Compare',
        compare_quality: 'Quality Comparison',
        iteration_history: 'Iteration History',
        btn_load_reports: 'Load Reports',
        my_models: 'My Models',
        btn_refresh: 'Refresh',
        btn_download: 'Download',
        btn_delete: 'Delete',
        trained: 'Trained',
        datasets_count: 'Datasets',
        size: 'Size',
        confirm_delete: 'Delete model',
        no_models: 'No models.',
        fill_name_url: 'Enter name and URL',
        wizard_welcome: 'Welcome!',
        wizard_desc: 'Create your first neural network in 3 simple steps.',
        btn_skip: 'Skip',
        preview: 'Preview',
        btn_close: 'Close',
        hw_recommendation: 'Recommendation',
        status_idle: 'Not running',
        checkpoint_a_label: 'Checkpoint A:',
        checkpoint_b_label: 'Checkpoint B:',
        compare_prompt_label: 'Prompt:',
        compare_default_prompt: 'Artificial intelligence',
        placeholder_name: 'Name',
        uploaded: 'Uploaded',
        btn_choose_file: 'Choose file',
        no_file_chosen: 'No file chosen',
        radar_generation: 'Generation',
        eta_sec: 's', eta_min: 'min', eta_h: 'h', eta_m: 'm',
        tab_autopilot: 'Autopilot',
        autopilot_title: 'LLM Autopilot',
        autopilot_desc: 'Describe your goal \u2014 AI will select data, create and train a model automatically',
        autopilot_provider_title: 'LLM Settings',
        autopilot_provider: 'Provider:',
        autopilot_model: 'Model:',
        autopilot_model_hint: 'gpt-4o',
        autopilot_api_key: 'API Key:',
        autopilot_endpoint: 'Endpoint:',
        autopilot_goal: 'Goal',
        autopilot_goal_placeholder: 'I want a model that writes detective stories like Sherlock Holmes in English',
        autopilot_start: 'Start Autopilot',
        autopilot_stop: 'Stop',
        autopilot_log: 'Action Log',
        autopilot_enter_goal: 'Enter a goal for the autopilot',
        autopilot_enter_key: 'Enter API key',
        autopilot_beta_notice: 'This feature is in beta testing. Results may be unstable. Requires an API key from OpenAI, Anthropic, or a local LLM.',
        autopilot_state_idle: 'Not running',
        autopilot_state_planning: 'Planning...',
        autopilot_state_executing: 'Executing actions...',
        autopilot_state_monitoring: 'Monitoring training...',
        autopilot_state_completed: 'Completed!',
        autopilot_state_error: 'Error',
        autopilot_state_stopped: 'Stopped',
        autopilot_time_budget: 'Time limit (minutes):',
        autopilot_time_hint: '0 = no limit. LLM will adjust iterations to fit the time.',
        autopilot_opt_google: 'Google AI Studio (Free)',
        autopilot_opt_groq: 'Groq (Free)',
        autopilot_opt_local: 'Local / Ollama / LM Studio',
        autopilot_help_google_ai: '<b>Google AI Studio — free Gemini (recommended):</b><br><br>1. Open <a href="https://aistudio.google.com/apikey" target="_blank" style="color:#667eea;text-decoration:underline;">aistudio.google.com/apikey</a> (sign in with Google)<br>2. Click <b>"Create API Key"</b><br>3. Copy the key<br>4. Paste it into the <b>API Key</b> field above<br><br>💡 <b>Completely free</b> — no credit card needed. Gemini 2.5 Flash is a powerful model with huge limits (1M tokens/min). Perfect for tool calling.',
        autopilot_help_groq: '<b>Groq — free and fast LLM:</b><br><br>1. Open <a href="https://console.groq.com/login" target="_blank" style="color:#667eea;text-decoration:underline;">console.groq.com</a> and sign in with Google (1 click)<br>2. Go to <a href="https://console.groq.com/keys" target="_blank" style="color:#667eea;text-decoration:underline;">API Keys</a><br>3. Click <b>"Create API Key"</b><br>4. Copy the key (starts with <code>gsk_...</code>)<br>5. Paste it into the <b>API Key</b> field above<br><br>⚠️ Free but 12K tokens/min limit — may cause pauses. If you get rate limit errors, try Google AI Studio instead.',
        autopilot_help_openai: '<b>OpenAI (paid):</b><br><br><b>How to get an API key:</b><br>1. Open <a href="https://platform.openai.com/signup" target="_blank" style="color:#667eea;text-decoration:underline;">platform.openai.com/signup</a> and create an account (or sign in with Google)<br>2. Open <a href="https://platform.openai.com/api-keys" target="_blank" style="color:#667eea;text-decoration:underline;">platform.openai.com/api-keys</a><br>3. Click the <b>"+ Create new secret key"</b> button<br>4. Copy the key (starts with <code>sk-...</code>) — it is shown only once!<br>5. Paste it into the <b>API Key</b> field above<br><br><b>⚠️ You need to add funds:</b> go to <a href="https://platform.openai.com/account/billing" target="_blank" style="color:#667eea;text-decoration:underline;">Billing</a> → Add payment method → add at least $5<br><br>💰 Cost: ~$0.01–0.05 per autopilot run (GPT-4o). $5 is enough for hundreds of runs.',
        autopilot_help_anthropic: '<b>Anthropic Claude (paid):</b><br><br><b>How to get an API key:</b><br>1. Open <a href="https://console.anthropic.com/login" target="_blank" style="color:#667eea;text-decoration:underline;">console.anthropic.com</a> and create an account<br>2. Open <a href="https://console.anthropic.com/settings/keys" target="_blank" style="color:#667eea;text-decoration:underline;">Settings → API Keys</a><br>3. Click <b>"Create Key"</b><br>4. Copy the key (starts with <code>sk-ant-...</code>) — it is shown only once!<br>5. Paste it into the <b>API Key</b> field above<br><br><b>⚠️ You need to add funds:</b> go to <a href="https://console.anthropic.com/settings/billing" target="_blank" style="color:#667eea;text-decoration:underline;">Billing</a> → Add payment method → add at least $5<br><br>💰 Cost: ~$0.01–0.05 per autopilot run. $5 is enough for hundreds of runs.',
        autopilot_help_openai_compatible: '<b>Local LLM (free, but uses your PC resources):</b><br><br>🔑 <b>No API key needed</b> — everything runs locally on your computer.<br><br><b>Option 1 — Ollama (easier):</b><br>1. Download and install from <a href="https://ollama.com/download" target="_blank" style="color:#667eea;text-decoration:underline;">ollama.com/download</a> (Windows/Mac/Linux)<br>2. Open a terminal and run:<br>&nbsp;&nbsp;&nbsp;<code>ollama pull llama3</code> — downloads the model (~4 GB)<br>3. Ollama starts automatically after installation<br>4. Endpoint: <code>http://localhost:11434/v1</code> (already filled in)<br>5. In the Model field type: <code>llama3</code><br><br><b>Option 2 — LM Studio (has a GUI):</b><br>1. Download from <a href="https://lmstudio.ai" target="_blank" style="color:#667eea;text-decoration:underline;">lmstudio.ai</a> and install<br>2. Search and download a model in the app (e.g. llama3)<br>3. Go to the <b>"Local Server"</b> tab (↔ icon on the left) → click <b>"Start Server"</b><br>4. Endpoint: <code>http://localhost:1234/v1</code><br>5. In the Model field enter the name of the downloaded model<br><br>⚠️ <b>Requires a powerful PC</b> (16GB+ RAM, GPU recommended). LLM runs alongside model training — may be slow. We recommend Google Colab if your PC is not powerful enough.',
        btn_attach: '+ Attach', btn_detach: 'Detach',
        filter_all: 'All',
        no_datasets: 'No datasets.', no_attached: 'No attached datasets.',
        downloading: 'Downloading...',
        already_downloaded: 'Downloaded',
        btn_download_ds: 'Download', btn_preview: 'Preview',
        searching: 'Searching...', nothing_found: 'Nothing found.',
        no_data: 'No data.',
        model_created: 'Model created! Parameters: ',
        enter_model_name: 'Enter model name',
        generating: 'Generating...',
        loading: 'Loading...',
        improvement: 'Reward improved by ',
        wh_start: 'Model is just starting. High loss is normal. Learning basic patterns.',
        wh_early: 'Model is learning frequent words and simple combinations.',
        wh_mid: 'Basic patterns learned. Reward is growing — generation quality improves.',
        wh_late: 'Training in progress. Model is catching subtle patterns. Loss decreasing slower is normal.',
        wh_final: 'Final stage! Last adjustments. Training will finish soon.',
        wh_done: 'Training is almost complete!',
        rl_diversity: 'Diversity', rl_coherence: 'Coherence', rl_repetition: 'No Repetition',
        rl_length: 'Length', rl_vocabulary: 'Vocabulary', rl_naturalness: 'Naturalness',
    }
};

// === AUTOPILOT ===
var autopilotPollInterval = null;
var autopilotLogSince = 0;

function updateProviderFields() {
    var p = document.getElementById('ap_provider').value;
    document.getElementById('ap_endpoint_group').style.display = (p === 'openai_compatible') ? 'block' : 'none';
    document.getElementById('ap_api_key_group').style.display = (p === 'openai_compatible') ? 'none' : 'block';
    var m = document.getElementById('ap_model');
    if (p === 'openai') m.placeholder = 'gpt-4o';
    else if (p === 'anthropic') m.placeholder = 'claude-sonnet-4-20250514';
    else if (p === 'google_ai') m.placeholder = 'gemini-2.5-flash';
    else if (p === 'groq') m.placeholder = 'llama-3.3-70b-versatile';
    else m.placeholder = 'llama3';
    var ep = document.getElementById('ap_endpoint');
    ep.placeholder = 'http://localhost:11434/v1';
    var help = document.getElementById('ap_provider_help');
    help.innerHTML = t('autopilot_help_' + p);
    help.style.display = 'block';
}
document.addEventListener('DOMContentLoaded', function() { updateProviderFields(); });

function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function startAutopilot() {
    var goal = document.getElementById('ap_goal').value.trim();
    if (!goal) { alert(t('autopilot_enter_goal')); return; }
    var apiKey = document.getElementById('ap_api_key').value.trim();
    var provider = document.getElementById('ap_provider').value;
    if (provider !== 'openai_compatible' && !apiKey) { alert(t('autopilot_enter_key')); return; }

    var sendProvider = provider;
    var sendEndpoint = document.getElementById('ap_endpoint').value || null;
    var sendModel = document.getElementById('ap_model').value || null;
    if (provider === 'google_ai') {
        sendProvider = 'openai_compatible';
        sendEndpoint = 'https://generativelanguage.googleapis.com/v1beta/openai';
        if (!sendModel) sendModel = 'gemini-2.5-flash';
    } else if (provider === 'groq') {
        sendProvider = 'openai_compatible';
        sendEndpoint = 'https://api.groq.com/openai/v1';
        if (!sendModel) sendModel = 'llama-3.3-70b-versatile';
    }

    var timeBudget = parseInt(document.getElementById('ap_time_budget').value) || 0;

    var body = {
        goal: goal,
        provider: sendProvider,
        api_key: apiKey || '',
        endpoint: sendEndpoint,
        model: sendModel,
        time_budget: timeBudget
    };

    fetch('/autopilot/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.status === 'started') {
            document.getElementById('ap_start_btn').style.display = 'none';
            document.getElementById('ap_stop_btn').style.display = 'inline-block';
            document.getElementById('ap_log_box').style.display = 'block';
            document.getElementById('ap_log_entries').innerHTML = '';
            document.getElementById('ap_state_box').style.display = 'block';
            autopilotLogSince = 0;
            autopilotPollInterval = setInterval(pollAutopilot, 2000);
        } else {
            alert(data.detail || 'Error starting autopilot');
        }
    })
    .catch(function(e) { alert('Error: ' + e.message); });
}

function pollAutopilot() {
    fetch('/autopilot/status?since=' + autopilotLogSince)
    .then(function(r) { return r.json(); })
    .then(function(data) {
        var stateEl = document.getElementById('ap_state_text');
        stateEl.textContent = t('autopilot_state_' + data.state);
        document.getElementById('ap_state_box').style.display = 'block';

        var container = document.getElementById('ap_log_entries');
        data.log.forEach(function(entry) {
            var div = document.createElement('div');
            div.className = 'ap-log-entry ap-log-' + entry.type;
            var timeStr = entry.timestamp ? entry.timestamp.substr(11, 8) : '';
            div.innerHTML = '<span class="ap-time">' + timeStr + '</span>' +
                '<span class="ap-type">' + entry.type + '</span>' +
                '<span class="ap-content">' + escapeHtml(entry.content) + '</span>';
            container.appendChild(div);
        });
        autopilotLogSince = data.log_count;
        container.scrollTop = container.scrollHeight;

        if (['completed','error','stopped','idle'].indexOf(data.state) !== -1) {
            clearInterval(autopilotPollInterval);
            autopilotPollInterval = null;
            document.getElementById('ap_start_btn').style.display = 'inline-block';
            document.getElementById('ap_stop_btn').style.display = 'none';
        }
    });
}

function stopAutopilot() {
    fetch('/autopilot/stop', {method:'POST'})
    .then(function(r) { return r.json(); });
}

function switchLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('azr_lang', lang);
    document.getElementById('lang_ru').classList.toggle('active', lang === 'ru');
    document.getElementById('lang_en').classList.toggle('active', lang === 'en');
    document.querySelectorAll('[data-i18n]').forEach(function(el) {
        var key = el.getAttribute('data-i18n');
        if (TRANSLATIONS[lang] && TRANSLATIONS[lang][key]) {
            el.textContent = TRANSLATIONS[lang][key];
        }
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function(el) {
        var key = el.getAttribute('data-i18n-placeholder');
        if (TRANSLATIONS[lang] && TRANSLATIONS[lang][key]) {
            el.placeholder = TRANSLATIONS[lang][key];
        }
    });
    document.querySelectorAll('[data-i18n-value]').forEach(function(el) {
        var key = el.getAttribute('data-i18n-value');
        if (TRANSLATIONS[lang] && TRANSLATIONS[lang][key]) {
            el.value = TRANSLATIONS[lang][key];
        }
    });
    // Update dynamic content using t()
    if (typeof REWARD_LABELS !== 'undefined') {
        REWARD_LABELS.diversity = t('rl_diversity');
        REWARD_LABELS.coherence = t('rl_coherence');
        REWARD_LABELS.repetition_penalty = t('rl_repetition');
        REWARD_LABELS.length_score = t('rl_length');
        REWARD_LABELS.vocabulary_richness = t('rl_vocabulary');
        REWARD_LABELS.bigram_naturalness = t('rl_naturalness');
    }
}

function t(key) {
    return (TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][key]) || key;
}

window.addEventListener('load', function() {
    loadModelsSelect();
    initChart();
    startStatusUpdates();
    checkWizard();
    // Restore saved language — always apply to set REWARD_LABELS etc.
    switchLanguage(currentLang);
});
</script>
</body>
</html>
'''

output_path = TEMPLATES_DIR / 'index_complete.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"SUCCESS: AZR Model Trainer v2 interface built!")
print(f"File: {output_path}")
print(f"Size: {output_path.stat().st_size // 1024} KB")
print(f"Run 'python server_with_datasets.py' to start")
