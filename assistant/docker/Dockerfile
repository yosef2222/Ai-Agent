FROM python:3.11-slim AS base

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid agent --no-create-home --shell /usr/sbin/nologin agent \
    && mkdir -p /workspace /app \
    && chown agent:agent /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app HOME=/workspace
WORKDIR /workspace

FROM base AS gateway
COPY docker/gateway.py /app/gateway.py
USER agent
CMD ["python", "/app/gateway.py"]

FROM base AS runner
RUN apt-get update && apt-get install -y --no-install-recommends libseccomp2 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/runner.py docker/snippet.py /app/
COPY agent/__init__.py agent/results.py /app/agent/
COPY agent/skills/__init__.py agent/skills/code_policy.py /app/agent/skills/
USER agent
CMD ["python", "/app/runner.py"]

FROM base AS assistant
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY agent/ /app/agent/
COPY autonomous/ /app/autonomous/
USER agent
CMD ["python", "-m", "agent.main"]
