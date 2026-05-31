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

# Diagnostic: confirm the LFS-tracked cache artifacts arrived as real
# binaries (not text pointer files) inside the build context.
RUN ls -lh cache/ && head -c 16 cache/pymc_idata.nc | od -An -c | head -1

# Regenerate the EconML pickles under the deployed numpy. They were
# created under numpy 1.26 locally and crash on load under numpy 1.23
# (numpy's __randomstate_ctor signature changed between those versions).
# The PyMC NetCDF posterior ships as-is — NetCDF is a portable binary
# format and the committed file loads fine across numpy versions.
# Skipping precompute_pymc() also avoids re-running the ~5 min MCMC and
# a separate pymc/numpy version conflict (pymc needs Generator.spawn).
RUN python -c "from precompute import precompute_econml; precompute_econml()"

EXPOSE 7860
CMD ["python", "-m", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
