#!/bin/bash

echo "🧹 Starting project cleanup..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Remove temporary files
find . -type f -name "*~" -delete
find . -type f -name ".DS_Store" -delete

# Remove log files
find . -type f -name "*.log" -delete

# Clean Docker volumes (optional - use with caution)
# docker-compose down -v

echo "✨ Cleanup completed!"
