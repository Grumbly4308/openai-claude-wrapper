# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim AS claude-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        ffmpeg \
        espeak-ng \
        imagemagick \
        librsvg2-bin \
        python3 \
        python3-pip \
        python3-venv \
        libmagic1 \
        ripgrep \
        tini \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code@latest \
    && claude --version

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN userdel -r node 2>/dev/null || true \
    && useradd --create-home --shell /bin/bash --uid 1000 claude \
    && mkdir -p /data/files /data/workspace /data/sessions /home/claude/.claude \
    && chown -R claude:claude /app /data /home/claude

USER claude

ENV CLAUDE_WRAPPER_DATA=/data \
    CLAUDE_WRAPPER_WORKSPACE=/data/workspace \
    CLAUDE_WRAPPER_FILES=/data/files \
    CLAUDE_WRAPPER_SESSIONS=/data/sessions \
    CLAUDE_WRAPPER_HOST=0.0.0.0 \
    CLAUDE_WRAPPER_PORT=8000

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
