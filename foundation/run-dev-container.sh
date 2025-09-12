#!/bin/bash

# ARES ChronoFabric Interactive Development Container Launcher
# Author: Ididia Serfaty

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="ares-dev"
IMAGE_NAME="ares-csf:dev-interactive-ubuntu"
DOCKERFILE="Dockerfile.dev-interactive"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  run         Run the interactive development container (default)"
    echo "  build       Build the container image"
    echo "  rebuild     Rebuild the container image from scratch"
    echo "  stop        Stop the running container"
    echo "  remove      Remove the container"
    echo "  clean       Stop and remove container, then remove image"
    echo "  shell       Get a shell in running container"
    echo "  logs        Show container logs"
    echo "  status      Show container status"
    echo ""
    echo "Options:"
    echo "  -h, --help  Show this help message"
    echo "  -v          Verbose output"
    echo ""
    echo "Environment Variables:"
    echo "  CLAUDE_API_KEY  Your Claude API key for full functionality"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run interactive container"
    echo "  $0 build             # Build container image"
    echo "  $0 run               # Run with current directory mounted"
    echo "  CLAUDE_API_KEY=sk-xxx $0 run  # Run with API key"
}

# Check if podman is available
check_podman() {
    if ! command -v podman >/dev/null 2>&1; then
        print_error "Podman is not installed or not in PATH"
        exit 1
    fi
}

# Build the container image
build_container() {
    local rebuild_flag=""
    if [[ "$1" == "rebuild" ]]; then
        rebuild_flag="--no-cache"
        print_status "Rebuilding container image from scratch..."
    else
        print_status "Building container image..."
    fi
    
    if [[ ! -f "$DOCKERFILE" ]]; then
        print_error "Dockerfile not found: $DOCKERFILE"
        exit 1
    fi
    
    print_status "Building image: $IMAGE_NAME"
    if podman build $rebuild_flag -f "$DOCKERFILE" -t "$IMAGE_NAME" .; then
        print_success "Container image built successfully"
    else
        print_error "Failed to build container image"
        exit 1
    fi
}

# Run the interactive container
run_container() {
    print_status "Starting interactive development container..."
    
    # Stop existing container if running
    if podman ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
        print_warning "Stopping existing container..."
        podman stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    
    # Remove existing container if it exists
    if podman ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
        print_warning "Removing existing container..."
        podman rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    
    # Prepare environment variables
    local env_args=""
    if [[ -n "$CLAUDE_API_KEY" ]]; then
        env_args="-e CLAUDE_API_KEY=$CLAUDE_API_KEY"
        print_status "Claude API key provided"
    else
        print_warning "No CLAUDE_API_KEY set - Claude functionality will be limited"
    fi
    
    # Get host user info for permission mapping
    local host_uid=$(id -u)
    local host_gid=$(id -g)
    
    # Run the container with proper permissions
    print_status "Launching container with workspace mounted..."
    print_status "Host UID:GID = $host_uid:$host_gid"
    exec podman run -it --rm \
        --name "$CONTAINER_NAME" \
        --privileged \
        $env_args \
        -v "$(pwd):/home/dev/workspace:z" \
        -v "claude-ubuntu-config:/home/dev/.claude" \
        -e HOST_UID="$host_uid" \
        -e HOST_GID="$host_gid" \
        --entrypoint "/bin/bash" \
        "$IMAGE_NAME" \
        -c "
        # Fix permissions for workspace access
        sudo chown -R dev:dev /home/dev/workspace 2>/dev/null || true
        sudo chmod -R u+rwX /home/dev/workspace 2>/dev/null || true
        
        # Start normal shell
        cd /home/dev/workspace
        exec /bin/bash -l
        "
}

# Stop the container
stop_container() {
    print_status "Stopping container..."
    if podman ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
        podman stop "$CONTAINER_NAME"
        print_success "Container stopped"
    else
        print_warning "Container is not running"
    fi
}

# Remove the container
remove_container() {
    print_status "Removing container..."
    if podman ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
        podman rm "$CONTAINER_NAME" 2>/dev/null || true
        print_success "Container removed"
    else
        print_warning "Container does not exist"
    fi
}

# Clean up everything
clean_all() {
    print_status "Cleaning up container and image..."
    stop_container
    remove_container
    
    if podman images -q "$IMAGE_NAME" | grep -q .; then
        podman rmi "$IMAGE_NAME" 2>/dev/null || true
        print_success "Image removed"
    else
        print_warning "Image does not exist"
    fi
}

# Get shell in running container
get_shell() {
    if podman ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
        print_status "Opening shell in running container..."
        exec podman exec -it "$CONTAINER_NAME" /bin/bash
    else
        print_error "Container is not running. Start it first with: $0 run"
        exit 1
    fi
}

# Show container logs
show_logs() {
    if podman ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
        podman logs "$CONTAINER_NAME"
    else
        print_error "Container does not exist"
        exit 1
    fi
}

# Show container status
show_status() {
    echo "Container Status:"
    echo "================="
    if podman ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep -q "$CONTAINER_NAME"; then
        podman ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    else
        echo "Container does not exist"
    fi
    
    echo ""
    echo "Image Status:"
    echo "============="
    if podman images -q "$IMAGE_NAME" | grep -q .; then
        podman images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.Created}}"
    else
        echo "Image does not exist"
    fi
}

# Main script logic
main() {
    check_podman
    
    # Parse command line arguments
    case "${1:-run}" in
        "run")
            # Check if image exists
            if ! podman images -q "$IMAGE_NAME" | grep -q .; then
                print_warning "Container image not found. Building it first..."
                build_container
            fi
            run_container
            ;;
        "build")
            build_container
            ;;
        "rebuild")
            build_container "rebuild"
            ;;
        "stop")
            stop_container
            ;;
        "remove")
            remove_container
            ;;
        "clean")
            clean_all
            ;;
        "shell")
            get_shell
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "-h"|"--help"|"help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run the main function
main "$@"