"""Load profile: mostly cheap classification/search, some LLM generation and chat (like a real UI session).

locust -f loadtest/locustfile.py --host http://localhost:7860 --headless -u 4 -r 1 -t 2m --csv report
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

RECIPES = [
    ["2 oz white rum", "1 lime", "10 fresh mint leaves", "2 tsp sugar", "club soda"],
    ["blackberries", "poblano peppers", "silver tequila", "cointreau"],
    ["1 lb spaghetti", "4 oz guanciale", "3 large eggs", "pecorino romano", "black pepper"],
    ["2 cups flour", "1 cup sugar", "2 eggs", "1 cup buttermilk", "2 ripe bananas"],
    ["butternut squash", "onion", "vegetable stock", "heavy cream", "nutmeg"],
    ["romaine lettuce", "parmesan", "croutons", "anchovies", "lemon juice"],
]
QUESTIONS = [
    "I have chicken, lemon and rice. What can I cook?",
    "Something sweet with apples",
    "gin and grapefruit drink",
]


class MenuUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(6)
    def classify(self) -> None:
        self.client.post("/v1/classify", json={"ingredients": random.choice(RECIPES)}, name="/v1/classify")

    @task(3)
    def search(self) -> None:
        self.client.post("/v1/search", json={"query": random.choice(QUESTIONS), "k": 5}, name="/v1/search")

    @task(2)
    def generate(self) -> None:
        self.client.post("/v1/generate", json={"ingredients": random.choice(RECIPES)}, name="/v1/generate", timeout=120)

    @task(1)
    def chat(self) -> None:
        self.client.post("/v1/chat", json={"message": random.choice(QUESTIONS)}, name="/v1/chat", timeout=120)

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
