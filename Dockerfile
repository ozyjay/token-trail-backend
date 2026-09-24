FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=/app/src \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); open('requirements.txt','w').write(chr(10).join(d['project']['dependencies'])+chr(10))" \
    && pip install --no-cache-dir -r requirements.txt
COPY scripts/download_model.py ./scripts/download_model.py
COPY src/token_trail_backend/config.py ./src/token_trail_backend/config.py
RUN mkdir -p /home/user/.cache/huggingface && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    python3 scripts/download_model.py
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
RUN useradd -m -u 1000 user && chown -R user:user /home/user /app
USER user
EXPOSE 7860
CMD ["uvicorn", "token_trail_backend.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--no-access-log"]
