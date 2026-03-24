#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH

# Load .env if present
if [ -f "$(dirname "$0")/.env" ]; then
  export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi