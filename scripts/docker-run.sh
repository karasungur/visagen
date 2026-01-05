#!/bin/bash
# =============================================================================
# Visagen Docker Run Script
# =============================================================================
# Runs Visagen Docker container with GPU support and proper volume mounts
#
# Usage:
#   ./scripts/docker-run.sh                    # Run Gradio UI
#   ./scripts/docker-run.sh --dev              # Run development mode
#   ./scripts/docker-run.sh --shell            # Open shell in container
#   ./scripts/docker-run.sh --train config.yml # Run training
#   ./scripts/docker-run.sh visagen-extract    # Run CLI command
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="visagen:latest"
CONTAINER_NAME="visagen"
WORKSPACE_DIR="${WORKSPACE_DIR:-./workspace}"

# Create workspace if not exists
mkdir -p "$WORKSPACE_DIR"

# Parse arguments
MODE="ui"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            IMAGE_NAME="visagen:dev"
            shift
            ;;
        --shell)
            MODE="shell"
            shift
            ;;
        --train)
            MODE="train"
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                EXTRA_ARGS="$2"
                shift
            fi
            shift
            ;;
        --jupyter)
            MODE="jupyter"
            IMAGE_NAME="visagen:dev"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [COMMAND]"
            echo ""
            echo "Options:"
            echo "  --dev       Run in development mode with source mounted"
            echo "  --shell     Open interactive shell in container"
            echo "  --train     Run training with optional config file"
            echo "  --jupyter   Start Jupyter notebook server"
            echo "  -h, --help  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Start Gradio UI"
            echo "  $0 --shell                # Open shell"
            echo "  $0 --train config.yml     # Run training"
            echo "  $0 visagen-export -h      # Run CLI command"
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
        *)
            # Treat remaining args as command
            MODE="cmd"
            EXTRA_ARGS="$*"
            break
            ;;
    esac
done

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Visagen Docker Run                        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Mode:${NC}      ${MODE}"
echo -e "${YELLOW}Image:${NC}     ${IMAGE_NAME}"
echo -e "${YELLOW}Workspace:${NC} ${WORKSPACE_DIR}"
echo ""

# Check if NVIDIA Docker runtime is available
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected.${NC}"
    echo "GPU acceleration may not be available."
    echo ""
    GPU_FLAG=""
else
    GPU_FLAG="--gpus all"
fi

# Base docker run command
DOCKER_CMD="docker run --rm -it \
    ${GPU_FLAG} \
    --name ${CONTAINER_NAME} \
    -v $(realpath ${WORKSPACE_DIR}):/workspace \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video"

case $MODE in
    ui)
        echo -e "${GREEN}Starting Gradio UI...${NC}"
        echo "Access at: http://localhost:7860"
        echo ""
        $DOCKER_CMD \
            -p 7860:7860 \
            ${IMAGE_NAME}
        ;;
    dev)
        echo -e "${GREEN}Starting development mode...${NC}"
        echo "Source code mounted for live editing"
        echo ""
        $DOCKER_CMD \
            -p 7860:7860 \
            -p 6006:6006 \
            -p 8888:8888 \
            -v $(pwd):/app \
            ${IMAGE_NAME} \
            /bin/bash
        ;;
    shell)
        echo -e "${GREEN}Opening shell...${NC}"
        $DOCKER_CMD \
            ${IMAGE_NAME} \
            /bin/bash
        ;;
    train)
        echo -e "${GREEN}Starting training...${NC}"
        $DOCKER_CMD \
            -p 6006:6006 \
            ${IMAGE_NAME} \
            python -m visagen.tools.train ${EXTRA_ARGS}
        ;;
    jupyter)
        echo -e "${GREEN}Starting Jupyter notebook...${NC}"
        echo "Access at: http://localhost:8888 (token: visagen)"
        echo ""
        $DOCKER_CMD \
            -p 8888:8888 \
            -e JUPYTER_TOKEN=visagen \
            ${IMAGE_NAME} \
            jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
        ;;
    cmd)
        echo -e "${GREEN}Running command: ${EXTRA_ARGS}${NC}"
        $DOCKER_CMD \
            ${IMAGE_NAME} \
            ${EXTRA_ARGS}
        ;;
esac
