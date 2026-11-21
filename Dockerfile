# Menu from Neural Networks — API + UI image (CPU). Works locally and as a Hugging Face Docker Space.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# WITH_LLM=0 builds a light image (classifier + search + UI; generation runs in retrieval-fallback mode)
ARG WITH_LLM=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# 1) dependencies (cached layer) from the lock file; CPU-only torch instead of the CUDA wheels
COPY pyproject.toml uv.lock README.md ./
RUN if [ "$WITH_LLM" = "1" ]; then EXTRA="--extra llm"; else EXTRA=""; fi \
 && uv export --frozen --no-dev $EXTRA --no-emit-project --no-hashes -o /tmp/req.txt \
 && grep -vE '^(torch|nvidia-|triton|cuda-)' /tmp/req.txt > /tmp/req-nocuda.txt \
 && if [ "$WITH_LLM" = "1" ]; then \
      grep -E '^torch==' /tmp/req.txt > /tmp/torch.txt && uv pip install -r /tmp/torch.txt --index-url "$TORCH_INDEX"; fi \
 && uv pip install -r /tmp/req-nocuda.txt \
 && rm -rf /root/.cache

# 2) application code
COPY src ./src
RUN uv pip install --no-deps . && python -c "import menu; print('menu', menu.__version__)"

# model cache owned by the runtime user (a named volume mounted here inherits this ownership)
RUN mkdir -p /home/user/.cache/huggingface /home/user/artifacts && chown -R user:user /home/user/.cache /home/user/artifacts
USER user
ENV MENU_ARTIFACTS_DIR=/home/user/artifacts PORT=7860 MENU_ENABLE_LLM=${WITH_LLM}
# Model bundle: mount ./artifacts or set MENU_HF_REPO (+ MENU_HF_REVISION) to download the pinned version at start.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health').status==200 else 1)"
CMD ["sh", "-c", "uvicorn menu.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
