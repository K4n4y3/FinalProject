"""ASGI entry point: ``uvicorn menu.api.main:app``."""

from menu.api.app import create_app

app = create_app()
