"""Screenshots of the running local stack (run inside the compose network, see screenshots_local.bat)."""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

APP, GRAFANA, PROM = "http://app:7860", "http://grafana:3000", "http://prometheus:9090"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/out")
OUT.mkdir(parents=True, exist_ok=True)
MARGARITA = "blackberries\npoblano peppers\nsilver tequila\ncointreau"


def tab(page, name):
    page.get_by_role("tab", name=name).click()
    time.sleep(0.8)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page.goto(f"{APP}/ui/", wait_until="load")
    page.get_by_role("button", name="Придумать блюдо").wait_for(timeout=60000)
    box = page.get_by_label("Ингредиенты").first
    box.fill(MARGARITA)
    page.get_by_role("button", name="Придумать блюдо").click()
    page.wait_for_function(
        "document.querySelector('#title-out') && document.querySelector('#title-out').innerText.length > 4",
        timeout=120000,
    )
    time.sleep(1)
    page.screenshot(path=OUT / "01_menu.png")
    print("menu ok")

    tab(page, "Сравнение моделей")
    page.get_by_role("button", name="Сравнить все генераторы").click()
    page.wait_for_function(
        "[...document.querySelectorAll('table')].some(t => t.innerText.includes('LoRA-RAG'))", timeout=240000
    )
    time.sleep(1)
    page.screenshot(path=OUT / "02_compare.png")
    print("compare ok")

    tab(page, "Чат-ассистент")
    page.get_by_role("button", name="I have chicken, lemon and rice. What can I cook?").first.click()
    page.wait_for_function("document.body.innerText.includes('Источники')", timeout=180000)
    time.sleep(1)
    page.screenshot(path=OUT / "03_chat.png")
    print("chat ok")

    tab(page, "Мониторинг")
    page.get_by_role("button", name="Обновить").click()
    time.sleep(2)
    page.screenshot(path=OUT / "04_monitoring.png")
    tab(page, "О модели")
    time.sleep(1)
    page.screenshot(path=OUT / "05_about.png", full_page=True)
    print("about ok")

    page.goto(f"{APP}/docs", wait_until="networkidle")
    time.sleep(1)
    page.screenshot(path=OUT / "06_swagger.png")

    page.goto(f"{GRAFANA}/d/menu-overview/?orgId=1&kiosk&from=now-30m&to=now", wait_until="networkidle")
    time.sleep(6)
    page.screenshot(path=OUT / "07_grafana.png")
    page.goto(f"{PROM}/alerts", wait_until="networkidle")
    time.sleep(2)
    page.screenshot(path=OUT / "08_prometheus_alerts.png")
    print("all screenshots saved to", OUT)
    b.close()
