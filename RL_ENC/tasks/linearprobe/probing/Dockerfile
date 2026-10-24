FROM rl-env-base

## TASK SPECIFIC CONTENT: Change to match your task

ARG USERNAME="model"

# Install task-specific requirements
RUN uv pip install --system huggingface_hub[cli]
RUN uv pip install --system torch transformers accelerate numpy scikit-learn tqdm
# Authenticate with Hugging Face
RUN hf auth login --token YOUR_HF_TOKEN_PLACEHOLDER

# Download model for task (as root -> /opt/huggingface, agent-readable/root-owned)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-270m')"
ENV YOUR_HF_TOKEN_PLACEHOLDER_OFFLINE=1

## GENERAL CONTENT: Should be the same for all tasks (place these at the end to play nicely with Docker build cache
# Copy task-specific grader code
COPY --chmod=700 grader /grader

# Copy task description
COPY --chmod=600 task.py /task.py
