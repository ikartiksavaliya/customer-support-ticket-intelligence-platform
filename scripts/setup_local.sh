#!/bin/bash
# ==============================================================================
# Local Environment Setup Script
# ==============================================================================
set -e

# Change directory to the repository root (where the script is located's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=================================================="
echo "🚀 Setting up local environment in: $(pwd)"
echo "=================================================="

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists in .venv."
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# 5. Install package in editable mode
echo "Installing project in editable mode..."
pip install -e .

# 6. Register Jupyter kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user \
  --name customer-support \
  --display-name "Python (.venv) Customer Support"

echo "=================================================="
echo "✅ Environment setup complete!"
echo "To activate your environment in the terminal, run:"
echo "    source .venv/bin/activate"
echo "=================================================="
