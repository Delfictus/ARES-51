#!/bin/bash
# ARES CSF Worktree Management Helper
# Helps manage Git worktrees for parallel feature development

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "ARES CSF Worktree Management Helper"
    echo ""
    echo "Usage: $0 COMMAND [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  list                    List all worktrees"
    echo "  create NAME BRANCH     Create new worktree with branch"
    echo "  remove NAME            Remove worktree"
    echo "  status                 Show status of all worktrees"
    echo "  sync NAME              Sync worktree with main branch"
    echo "  build NAME             Build specific worktree"
    echo "  test NAME              Run tests in specific worktree"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 create consensus-impl feature/consensus-implementation"
    echo "  $0 sync ares-parallel-dev"
    echo "  $0 build ares-parallel-dev"
    echo "  $0 test ares-parallel-dev"
}

list_worktrees() {
    echo -e "${BLUE}Current worktrees:${NC}"
    git worktree list
}

create_worktree() {
    local name="$1"
    local branch="$2"
    local worktree_path="../$name"
    
    if [[ -z "$name" || -z "$branch" ]]; then
        echo -e "${RED}Error: Name and branch are required${NC}"
        usage
        exit 1
    fi
    
    echo -e "${YELLOW}Creating worktree '$name' with branch '$branch'...${NC}"
    
    if [[ -d "$worktree_path" ]]; then
        echo -e "${RED}Error: Directory $worktree_path already exists${NC}"
        exit 1
    fi
    
    git worktree add "$worktree_path" -b "$branch"
    
    echo -e "${GREEN}✅ Worktree created successfully!${NC}"
    echo -e "${BLUE}Location: $worktree_path${NC}"
    echo -e "${BLUE}Branch: $branch${NC}"
    
    # Copy CLAUDE.md to new worktree for development context
    cp CLAUDE.md "$worktree_path/CLAUDE.md"
    echo -e "${GREEN}✅ CLAUDE.md copied to new worktree${NC}"
}

remove_worktree() {
    local name="$1"
    local worktree_path="../$name"
    
    if [[ -z "$name" ]]; then
        echo -e "${RED}Error: Name is required${NC}"
        usage
        exit 1
    fi
    
    if [[ ! -d "$worktree_path" ]]; then
        echo -e "${RED}Error: Worktree $worktree_path does not exist${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Removing worktree '$name'...${NC}"
    
    # Get branch name before removing
    local branch=$(cd "$worktree_path" && git branch --show-current)
    
    git worktree remove "$worktree_path" --force
    
    # Ask if user wants to delete the branch
    read -p "Delete branch '$branch'? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -D "$branch" || true
        echo -e "${GREEN}✅ Branch '$branch' deleted${NC}"
    fi
    
    echo -e "${GREEN}✅ Worktree removed successfully!${NC}"
}

show_status() {
    echo -e "${BLUE}Worktree Status:${NC}"
    echo ""
    
    git worktree list | while read -r path commit branch; do
        echo -e "${YELLOW}Worktree: $path${NC}"
        echo -e "${BLUE}Branch: $branch${NC}"
        echo -e "${BLUE}Commit: $commit${NC}"
        
        if [[ -d "$path" ]]; then
            cd "$path"
            local status=$(git status --porcelain)
            if [[ -n "$status" ]]; then
                echo -e "${YELLOW}Changes:${NC}"
                echo "$status"
            else
                echo -e "${GREEN}Clean working directory${NC}"
            fi
        fi
        echo ""
    done
}

sync_worktree() {
    local name="$1"
    local worktree_path="../$name"
    
    if [[ -z "$name" ]]; then
        echo -e "${RED}Error: Name is required${NC}"
        usage
        exit 1
    fi
    
    if [[ ! -d "$worktree_path" ]]; then
        echo -e "${RED}Error: Worktree $worktree_path does not exist${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Syncing worktree '$name' with main branch...${NC}"
    
    # Fetch latest changes
    git fetch origin
    
    cd "$worktree_path"
    local current_branch=$(git branch --show-current)
    
    # Rebase current branch on main
    echo -e "${BLUE}Rebasing $current_branch on origin/main...${NC}"
    git rebase origin/main
    
    echo -e "${GREEN}✅ Worktree synced successfully!${NC}"
}

build_worktree() {
    local name="$1"
    local worktree_path="../$name"
    
    if [[ -z "$name" ]]; then
        echo -e "${RED}Error: Name is required${NC}"
        usage
        exit 1
    fi
    
    if [[ ! -d "$worktree_path" ]]; then
        echo -e "${RED}Error: Worktree $worktree_path does not exist${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Building worktree '$name'...${NC}"
    
    cd "$worktree_path"
    
    # Build all crates
    cargo build --all
    
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
}

test_worktree() {
    local name="$1"
    local worktree_path="../$name"
    
    if [[ -z "$name" ]]; then
        echo -e "${RED}Error: Name is required${NC}"
        usage
        exit 1
    fi
    
    if [[ ! -d "$worktree_path" ]]; then
        echo -e "${RED}Error: Worktree $worktree_path does not exist${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Running tests in worktree '$name'...${NC}"
    
    cd "$worktree_path"
    
    # Run all tests
    cargo test --all
    
    echo -e "${GREEN}✅ All tests passed!${NC}"
}

# Change to repository root
cd "$REPO_ROOT"

# Parse command
case "${1:-}" in
    list)
        list_worktrees
        ;;
    create)
        create_worktree "${2:-}" "${3:-}"
        ;;
    remove)
        remove_worktree "${2:-}"
        ;;
    status)
        show_status
        ;;
    sync)
        sync_worktree "${2:-}"
        ;;
    build)
        build_worktree "${2:-}"
        ;;
    test)
        test_worktree "${2:-}"
        ;;
    *)
        usage
        exit 1
        ;;
esac