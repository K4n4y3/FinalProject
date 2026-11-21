# Мониторинг

Три уровня: **сервис** (жив ли и быстро ли отвечает), **модель** (что и насколько уверенно предсказывает),
**данные** (похожи ли новые входы на обучающие).

## Метрики Prometheus (`/metrics`)

| метрика | тип | уровень | зачем |
|---|---|---|---|
| `menu_requests_total{endpoint,status}` | counter | сервис | нагрузка, доля ошибок и невалидных запросов |
| `menu_request_latency_seconds{endpoint}` | histogram | сервис | p50/p95 задержки |
| `menu_model_latency_seconds{model}` | histogram | модель | время классификатора и LLM отдельно |
| `menu_model_loaded{component}` | gauge | сервис | загружены ли classifier / retriever / generator / dense_encoder |
| `menu_predicted_class_total{label}` | counter | модель | сдвиг распределения предсказанных классов |
| `menu_prediction_confidence` | histogram | модель | падение уверенности — ранний признак дрейфа |
| `menu_low_confidence_total` | counter | модель | доля неоднозначных входов |
| `menu_input_oov_rate` | histogram | данные | доля слов ввода, неизвестных модели |
| `menu_drift_alert` | gauge | данные | 1, если средний OOV в окне > baseline validation + 0.15 |
| `menu_degraded_generations_total` | counter | модель | ответы fallback без LLM |
| `menu_rag_context_copies_total` | counter | модель | названия, скопированные из найденных рецептов |
| `menu_build_info{bundle,package}` | info | — | какая версия модели обслуживает трафик |

**Baseline дрейфа** считается при сборке бандла (средний OOV на validation, ≈0.008) и хранится в манифесте.
Пример: ввод на русском даёт OOV ≈ 1.0 → предупреждение в ответе, а при устойчивом потоке — `menu_drift_alert = 1`.

## Где смотреть

* **UI → вкладка «Мониторинг»** — скользящее окно последних 500 запросов и состояние компонентов (обновление 10 с).
* **`GET /v1/monitoring`** — то же в JSON.
* **Grafana** (`docker compose up`) — дашборд *Menu from Neural Networks — service & model*: RPS по эндпоинтам,
  ошибки, p50/p95, задержка моделей, классы, уверенность, OOV/дрейф, загруженные компоненты, деградация.
* **Логи** — JSON в stdout (на HF Space — вкладка *Logs*): `request_id`, путь, статус, длительность.
* **Журнал предсказаний** — `MENU_PREDICTION_LOG=/path/preds.jsonl` (для офлайн-анализа и дообучения).

## Алерты (`monitoring/alerts.yml`)

| алерт | условие | реакция |
|---|---|---|
| HighErrorRate | >5 % ошибок 5 мин | логи по `request_id`, откат версии |
| SlowGeneration | p95 LLM > 8 с 10 мин | больше потоков / меньше `num_beams` / масштабирование |
| InputDrift | `menu_drift_alert == 1` 15 мин | проанализировать журнал предсказаний, дообучить |
| LowConfidenceSpike | >30 % неуверенных 30 мин | проверить новые типы входов |
| GeneratorDegraded | LLM не загружена 10 мин | проверить логи загрузки, доступ к HF Hub |
