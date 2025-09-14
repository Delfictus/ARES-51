#!/bin/bash
echo "Testing benchmark-suite compilation..."
cd /media/diddy/ARES-51/CAPO-AI/prct-engine
timeout 60 cargo check --bin benchmark-suite 2>&1 | head -50
echo "Done."