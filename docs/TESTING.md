# Тестирование

```bash
uv run pytest --cov            # все тесты (~40 с, без сети)
uv run pytest -m "not data"    # как в CI
uv run pytest -m data          # регрессия на реальных данных (нужен data/epicurious_clean_split.csv)
```

## Пирамида тестов

| уровень | файл | что проверяется |
|---|---|---|
| Unit | `test_text_and_data.py` | `clean_ingredient`, разбор ввода, правила типа блюда, очистка, union-find, детектор утечки |
| Unit | `test_models.py` | ROUGE-L, метрики генерации и `context_copy`, классификатор (обучение, калибровка, OOV, save/load), BM25, retriever (исключение групп, гибридный поиск, сериализация), отсутствие цели в RAG-примерах, разбор ответа LLM, fallback, целостность бандла |
| Интеграционные / контракт API | `test_api.py` | `/health`, `/v1/model`, схема ответов, 5 режимов генерации на крошечной Qwen2 с двумя LoRA-адаптерами, `/v1/menu`, поиск, чат, 422 на невалидный ввод, предупреждение о неизвестных словах, `/metrics`, монтирование UI, **деградация без LLM**, ключ API |
| Регрессия | `test_regression_real_data.py` | размеры разбиения и отсутствие утечки; macro-F1 0.710 A4 воспроизводится; финальный классификатор не хуже на обоих форматах ввода; ROUGE-L 0.386 / 0.374 / 0.146 из A4 пересчитываются тем же кодом |
| Smoke контейнера | CI job `docker` | сборка образа, запуск, `/health`, `/v1/classify`, `/metrics`, `/ui` |
| Smoke обучения | `notebooks/final_training.py` с `FINAL_OVERRIDES` | весь пайплайн обучения на CPU с крошечной моделью |
| Нагрузка | `loadtest/locustfile.py` | профиль UI-сессии; локально и после каждого деплоя |

Тесты не ходят в сеть: данные синтетические (`tests/conftest.py::make_split`), LLM — случайная 2-слойная Qwen2 с
обученным на лету BPE-токенизатором и chat-template (`tests/tiny_model.py`). Это проверяет **весь путь кода**
(PEFT multi-adapter, chat template, beam search, разбор ответа), не завися от 1 ГБ весов.

## Качество кода

* `ruff` (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, pylint-правила) + `ruff format`;
* `mypy` для `src/menu`;
* `pre-commit`: пробелы, YAML/TOML, большие файлы, приватные ключи, ruff, синхронность ноутбука с исходником.
