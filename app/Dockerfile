FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6 libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user server.py    .
COPY --chown=user pcg_glue     ./pcg_glue
COPY --chown=user static       ./static
COPY --chown=user demo_data    ./demo_data

ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
