FROM rl-env-base

## TASK SPECIFIC CONTENT: Change to match your task

# No additional task-specific packages needed for simple_math
# For tasks that need packages, install them here with:
# RUN uv pip install --system numpy pandas torch

## GENERAL CONTENT: Should be the same for all tasks (place these at the end to play nicely with Docker build cache)
# Copy task-specific grader code
COPY --chmod=700 grader /grader

# Copy task description
COPY --chmod=600 TASK.txt /TASK.txt
