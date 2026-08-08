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

# The in-container `claude` user's uid/gid MUST match the host user that owns
# any bind-mounted paths (the inbox, a shared credentials file), otherwise the
# container reads them as a stranger and 600-mode files are simply unreadable.
# Override at build time to match the host account, e.g. for a host `claude`
# at uid 1001:  docker compose build --build-arg CLAUDE_UID=1001 --build-arg CLAUDE_GID=1001
# (docker-compose.yml wires these to CLAUDE_UID/CLAUDE_GID in .env for you.)
ARG CLAUDE_UID=1000
ARG CLAUDE_GID=1000

# `node` occupies uid 1000 in the base image; remove it so that uid is free.
RUN userdel -r node 2>/dev/null || true \
    && (getent group "${CLAUDE_GID}" >/dev/null || groupadd --gid "${CLAUDE_GID}" claude) \
    && useradd --create-home --shell /bin/bash \
        --uid "${CLAUDE_UID}" --gid "${CLAUDE_GID}" claude \
    && mkdir -p /data/files /data/workspace /data/sessions /home/claude/.claude \
    && chown -R "${CLAUDE_UID}:${CLAUDE_GID}" /app /data /home/claude

USER claude

# By default the CLI splits its state either side of the volume boundary:
# ~/.claude/ holds credentials, projects and sessions, but ~/.claude.json —
# the config, including which directories are trusted — sits at HOME root.
# Only the directory is on the claude-home volume, so `run --rm` logins wrote
# the config to the container layer and lost it on exit, leaving the CLI to
# report a missing config next to a backup that did persist. Pointing
# CLAUDE_CONFIG_DIR at the mounted directory puts all of it on the volume.
ENV CLAUDE_CONFIG_DIR=/home/claude/.claude \
    CLAUDE_WRAPPER_DATA=/data \
    CLAUDE_WRAPPER_WORKSPACE=/data/workspace \
    CLAUDE_WRAPPER_FILES=/data/files \
    CLAUDE_WRAPPER_SESSIONS=/data/sessions \
    CLAUDE_WRAPPER_HOST=0.0.0.0 \
    CLAUDE_WRAPPER_PORT=8000

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
