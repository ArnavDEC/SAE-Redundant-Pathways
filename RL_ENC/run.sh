#!/usr/bin/env bash
# Thin wrapper around Python-based task runner
# Provides backward compatibility with original run.sh interface

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Setup network isolation for docker containers
# The reason we want to do this is to avoid any of the complexities or edge case failures that could result from the model e.g, downloading stuff, installing new packages, etc.
# (This puts the burden on the user to specify dependencies ahead of time, and also of course means our tasks aren't testing capabilities that involve e.g web search or package management.)
# However, in order for Claude Code to run at all, of course we still need network access to the Anthropic API.
setup_network_isolation() {
    local NETWORK_NAME="anthropic_only"
    local ANTHROPIC_API_IPS="160.79.104.0/23"

    # Create Docker network if it doesn't exist
    if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
        echo "Creating $NETWORK_NAME Docker network..."
        docker network create "$NETWORK_NAME"
    fi

    # Get the bridge interface name from Docker
    # Docker names custom bridges as br-<network_id_prefix>
    NETWORK_ID=$(docker network inspect "$NETWORK_NAME" -f '{{.Id}}')
    BRIDGE_INTERFACE="br-${NETWORK_ID:0:12}"

    # Detect if we're on macOS
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "Info: macOS detected - configuring iptables inside Docker Desktop VM..."
        # On macOS, Docker runs in a Linux VM. We need to configure iptables inside that VM.
        # We'll use a privileged container to access the VM's network namespace

        # Check if ACCEPT rule already exists by listing rules and grepping
        # iptables -S outputs flags in different order, so check for both interface and destination
        if ! docker run --rm --privileged --net=host --pid=host alpine:latest \
            nsenter -t 1 -m -u -n -i iptables -S DOCKER-USER | grep -q "DOCKER-USER.*-d $ANTHROPIC_API_IPS.*-i $BRIDGE_INTERFACE.*-j ACCEPT"; then
            echo "Adding iptables ACCEPT rule for Anthropic API on $BRIDGE_INTERFACE..."
            # Allow traffic to Anthropic API (insert at beginning)
            docker run --rm --privileged --net=host --pid=host alpine:latest \
                nsenter -t 1 -m -u -n -i iptables -I DOCKER-USER 1 -i "$BRIDGE_INTERFACE" -d "$ANTHROPIC_API_IPS" -j ACCEPT
        else
            echo "ACCEPT rule already exists for $BRIDGE_INTERFACE"
        fi

        # Check if REJECT rule already exists
        if ! docker run --rm --privileged --net=host --pid=host alpine:latest \
            nsenter -t 1 -m -u -n -i iptables -S DOCKER-USER | grep -q "DOCKER-USER.*-i $BRIDGE_INTERFACE.*-j REJECT"; then
            echo "Adding iptables REJECT rule for $BRIDGE_INTERFACE..."
            # Block all other outbound traffic from this network (insert at position 2, before RETURN)
            docker run --rm --privileged --net=host --pid=host alpine:latest \
                nsenter -t 1 -m -u -n -i iptables -I DOCKER-USER 2 -i "$BRIDGE_INTERFACE" -j REJECT
        else
            echo "REJECT rule already exists for $BRIDGE_INTERFACE"
        fi
    else
        # Linux: Use iptables directly on host
        # Check if ACCEPT rule already exists
        if ! sudo iptables -C DOCKER-USER -i "$BRIDGE_INTERFACE" -d "$ANTHROPIC_API_IPS" -j ACCEPT 2>/dev/null; then
            echo "Adding iptables rules for Anthropic API isolation on $BRIDGE_INTERFACE..."
            # Allow traffic to Anthropic API (insert at beginning)
            sudo iptables -I DOCKER-USER 1 -i "$BRIDGE_INTERFACE" -d "$ANTHROPIC_API_IPS" -j ACCEPT
        fi

        # Check if DROP rule already exists
        if ! sudo iptables -C DOCKER-USER -i "$BRIDGE_INTERFACE" -j REJECT 2>/dev/null; then
            # Block all other outbound traffic from this network (insert at position 2, before RETURN)
            sudo iptables -I DOCKER-USER 2 -i "$BRIDGE_INTERFACE" -j REJECT
        fi

        # TPU support: Allow access to GCE metadata server and TPU ports
        # GCE metadata server (required by torch_xla to initialize TPU)
        if ! sudo iptables -C DOCKER-USER -i "$BRIDGE_INTERFACE" -d 169.254.169.254 -p tcp --dport 80 -j ACCEPT 2>/dev/null; then
            echo "Adding iptables rule for GCE metadata server access..."
            sudo iptables -I DOCKER-USER 1 -i "$BRIDGE_INTERFACE" -d 169.254.169.254 -p tcp --dport 80 -j ACCEPT
        fi

        # Get Docker gateway IP for TPU access
        GATEWAY_IP=$(docker network inspect "$NETWORK_NAME" -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || echo "")
        if [ -n "$GATEWAY_IP" ]; then
            if ! sudo iptables -C DOCKER-USER -i "$BRIDGE_INTERFACE" -d "$GATEWAY_IP" -p tcp -m multiport --dports 8470,9230 -j ACCEPT 2>/dev/null; then
                echo "Adding iptables rule for TPU access on $GATEWAY_IP..."
                sudo iptables -I DOCKER-USER 1 -i "$BRIDGE_INTERFACE" -d "$GATEWAY_IP" -p tcp -m multiport --dports 8470,9230 -j ACCEPT
            fi
        fi
    fi
}
setup_network_isolation

# Validate GPU setup if per-container mode is requested
setup_gpu_isolation() {
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Warning: nvidia-smi not found. GPU passthrough will not be available."
        return
    fi

    # Check that NVIDIA Container Toolkit runtime is installed
    if ! docker info 2>/dev/null | grep -qi nvidia; then
        echo "Error: NVIDIA Container Toolkit runtime not detected."
        echo "GPU passthrough requires the NVIDIA Container Toolkit."
        echo "Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi

    # If --gpu-mode per-container is in the arguments, show discovered GPUs
    for arg in "$@"; do
        if [ "$arg" = "per-container" ]; then
            echo "Discovered GPU/MIG devices:"
            nvidia-smi -L
            echo ""
            break
        fi
    done
}

# Only run GPU validation if nvidia-smi exists and GPU flags are present
for arg in "$@"; do
    if [ "$arg" = "--gpu-mode" ]; then
        setup_gpu_isolation "$@"
        break
    fi
done

# Forward all arguments to Python script
exec python3 "$SCRIPT_DIR/tools/task_runner.py" run "$@"
