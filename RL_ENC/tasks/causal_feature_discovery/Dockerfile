FROM rl-env-base

RUN uv pip install --system --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple "torch==2.6.0+cpu"

RUN mkdir -p /opt/causal-feature-lab && chmod 700 /opt/causal-feature-lab

COPY --chmod=700 grader /grader
RUN OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 /grader/build_sae_artifact.py /opt/causal-feature-lab/sae_state.pt

COPY --chmod=700 experiment_server.py /opt/causal-feature-lab/server.py
COPY --chmod=755 experiment.py /usr/local/bin/experiment
COPY --chmod=600 TASK.py /TASK.py
