---
title: Menu from Neural Networks
emoji: 🍽️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Dish type, dish name and recipe RAG from ingredients
---

# Menu from Neural Networks

From a list of ingredients the system predicts the **dish type**, invents a **dish name** (Qwen2.5-0.5B + LoRA,
optionally with retrieved recipes — RAG) and answers **"what can I cook?"** grounded in Epicurious recipes.

* UI: `/ui` · REST API docs: `/docs` · health: `/health` · Prometheus metrics: `/metrics`
* Model bundle (pinned revision): [{{MODEL_REPO}}](https://huggingface.co/{{MODEL_REPO}})

Source code, tests and documentation: https://github.com/K4n4y3/FinalProject
