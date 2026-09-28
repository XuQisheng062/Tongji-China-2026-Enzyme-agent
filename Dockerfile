FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install .
COPY experiments ./experiments
COPY tests ./tests
COPY config/request.example.json config/bioartifact.example.json ./config/
COPY config/benchmark_tasks.example.json ./config/
COPY docs ./docs
CMD ["python", "experiments/run_verifier_reflection_ablation.py", "--mode", "offline", "--max-tasks", "2", "--output-dir", "/results/smoke"]
