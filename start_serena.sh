#!/bin/bash
set -e

# --- Configuration ---
PROJECT_DIR="/media/diddy/ARES-51/CAPO-AI"
API_KEY="ares512025"
SERVER_PORT="8080"
CONFIG_FILE="serena_config.yml"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}--- Serena Server Launcher (Final Version) ---${NC}"

# Navigate to the project directory first
cd "$PROJECT_DIR"

# Check for the config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found in $(pwd)${NC}"
    echo -e "${YELLOW}Please create it before running this script.${NC}"
    exit 1
fi

echo "This script will start the Serena server for project: $(pwd)"
echo -e "\n${YELLOW}Press [Enter] to continue...${NC}"
read

echo -e "\n${GREEN}Launching server...${NC}\n"

# The final, correct command based on the --help output
uvx mcpo --port "$SERVER_PORT" --api-key "$API_KEY" --config "$CONFIG_FILE"
