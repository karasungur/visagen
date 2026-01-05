#!/bin/bash
# =============================================================================
# Visagen Docker Build Script
# =============================================================================
# Builds Docker images for Visagen with GPU support
#
# Usage:
#   ./scripts/docker-build.sh              # Build runtime image
#   ./scripts/docker-build.sh --dev        # Build development image
#   ./scripts/docker-build.sh --all        # Build all images
#   ./scripts/docker-build.sh --no-cache   # Build without cache
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="visagen"
VERSION=$(grep -oP 'version = "\K[^"]+' pyproject.toml 2>/dev/null || echo "latest")

# Parse arguments
BUILD_DEV=false
BUILD_ALL=false
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            BUILD_DEV=true
            shift
            ;;
        --all)
            BUILD_ALL=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev       Build development image only"
            echo "  --all       Build all images (runtime + dev)"
            echo "  --no-cache  Build without Docker cache"
            echo "  -h, --help  Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   Visagen Docker Build                        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Version:${NC} ${VERSION}"
echo -e "${YELLOW}Image:${NC}   ${IMAGE_NAME}"
echo ""

# Build runtime image
if [ "$BUILD_DEV" = false ] || [ "$BUILD_ALL" = true ]; then
    echo -e "${GREEN}Building runtime image...${NC}"
    docker build \
        $NO_CACHE \
        --target runtime \
        -t "${IMAGE_NAME}:${VERSION}" \
        -t "${IMAGE_NAME}:latest" \
        -f Dockerfile \
        .
    echo -e "${GREEN}✓ Runtime image built: ${IMAGE_NAME}:${VERSION}${NC}"
    echo ""
fi

# Build development image
if [ "$BUILD_DEV" = true ] || [ "$BUILD_ALL" = true ]; then
    echo -e "${GREEN}Building development image...${NC}"
    docker build \
        $NO_CACHE \
        --target development \
        -t "${IMAGE_NAME}:dev" \
        -t "${IMAGE_NAME}:development" \
        -f Dockerfile \
        .
    echo -e "${GREEN}✓ Development image built: ${IMAGE_NAME}:dev${NC}"
    echo ""
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Available images:"
docker images | grep "${IMAGE_NAME}" | head -10
echo ""
echo -e "${YELLOW}To run:${NC}"
echo "  docker run --gpus all -p 7860:7860 ${IMAGE_NAME}:latest"
echo ""
echo -e "${YELLOW}Or use docker compose:${NC}"
echo "  docker compose up"
