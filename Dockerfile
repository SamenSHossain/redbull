FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        gfortran \
        graphviz \
        libgraphviz-dev \
        pkg-config \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "-m", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
