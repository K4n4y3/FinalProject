# REST API

Интерактивная документация (OpenAPI 3) — `/docs` (Swagger UI) и `/redoc`. Все ответы JSON, все ответы
инференса содержат `meta.model_version` и `meta.latency_ms`, каждый ответ — заголовок `X-Request-ID`.

| метод | путь | описание |
|---|---|---|
| GET | `/health` | состояние компонентов: `loading` → `ok` / `degraded` |
| GET | `/metrics` | метрики Prometheus |
| GET | `/v1/model` | версии, финальный вариант генератора, JSON-схемы входа и выхода |
| GET | `/v1/monitoring` | скользящее окно: задержки, распределение классов, уверенность, дрейф |
| POST | `/v1/classify` | тип блюда + top-k вероятностей |
| POST | `/v1/generate` | название блюда (`mode`: auto, lora, lora_rag, zero_shot, zero_shot_rag) |
| POST | `/v1/menu` | классификация + название одним запросом |
| POST | `/v1/search` | похожие рецепты (BM25 + плотный поиск) |
| POST | `/v1/chat` | ответ ассистента по найденным рецептам, с источниками |

## Примеры

```bash
curl -s -X POST $HOST/v1/classify -H 'content-type: application/json' \
  -d '{"ingredients": "blackberries; poblano peppers; silver tequila; cointreau", "top_k": 3}'
```

```json
{
  "dish_type": "Drink",
  "confidence": 0.9412,
  "top": [{"label": "Drink", "probability": 0.9412}, {"label": "Sauce & Condiment", "probability": 0.0213}, "..."],
  "low_confidence": false,
  "normalized_ingredients": ["blackberries", "poblano peppers", "silver tequila", "cointreau"],
  "meta": {"model_version": "1.0.0", "latency_ms": 2.7, "warnings": []}
}
```

```bash
curl -s -X POST $HOST/v1/generate -H 'content-type: application/json' \
  -d '{"ingredients": ["blackberries", "poblano peppers", "silver tequila", "cointreau"], "mode": "lora_rag"}'
```

```json
{
  "title": "Blackberry-Poblano Margarita",
  "valid": true,
  "mode": "lora_rag",
  "degraded": false,
  "context": [{"recipe_id": 1234, "title": "...", "ingredients": "...", "course": "Drink", "score": 14.2}],
  "context_copy": false,
  "meta": {"model_version": "1.0.0", "latency_ms": 1350.0, "warnings": []}
}
```

```bash
curl -s -X POST $HOST/v1/chat -H 'content-type: application/json' \
  -d '{"message": "I have chicken, lemon and rice. What can I cook?"}'
```

Из Python:

```python
import httpx

r = httpx.post(f"{HOST}/v1/menu", json={"ingredients": ["2 oz white rum", "1 lime", "mint", "sugar", "club soda"]})
r.raise_for_status()
print(r.json()["classification"]["dish_type"], r.json()["generation"]["title"])
```

## Ошибки

| код | когда |
|---|---|
| 401 | задан `MENU_API_KEY`, а заголовок `X-API-Key` отсутствует или неверен |
| 422 | невалидный ввод: нет ингредиентов, нет букв, `top_k` вне 1–9, слишком длинный текст |
| 503 | бандл ещё загружается |
| 500 | внутренняя ошибка (ответ содержит `request_id` для поиска в логах) |

Если LLM недоступна, `/v1/generate` и `/v1/chat` **не падают**: возвращают `degraded: true` и ответ на основе
найденных рецептов; `/health` показывает `degraded`.

## Конфигурация (переменные окружения)

| переменная | по умолчанию | смысл |
|---|---|---|
| `MENU_ARTIFACTS_DIR` | `artifacts` | каталог бандла |
| `MENU_HF_REPO`, `MENU_HF_REVISION` | — | откуда скачать бандл, если каталога нет (закреплённая ревизия) |
| `MENU_ENABLE_LLM` / `MENU_ENABLE_DENSE` | `true` | загрузка Qwen / MiniLM |
| `MENU_LLM_THREADS` | авто | потоки torch на CPU |
| `MENU_API_KEY` | — | защита `/v1/*` ключом |
| `MENU_PREDICTION_LOG` | — | путь JSONL-журнала предсказаний |
| `MENU_LOW_CONFIDENCE`, `MENU_DRIFT_OOV_MARGIN` | 0.4, 0.15 | пороги мониторинга |
