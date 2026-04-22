FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements/requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

ENV PYTHONPATH=/workspace

CMD ["python", "-m", "scripts.run_full_experiment_v2", "--help"]