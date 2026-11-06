#!/usr/bin/env bash
#
# Create a conda environment called "myenv" and install the required libraries
# in an order that respects their dependencies.
#
# Install order:
#   1. conda-forge (compiled scientific stack):
#        numpy -> scipy -> pandas -> scikit-learn -> matplotlib
#   2. pip (pure-Python packages that build on the stack above):
#        gower (needs numpy, scipy, scikit-learn)
#        plotly (needed by dash)
#        dash   (needs plotly)
#        anthropic
 
set -euo pipefail
 
ENV_NAME="myenv"
PYTHON_VERSION="3.11"
 
# Step 1: installed through conda, in dependency order
CONDA_PACKAGES=(numpy scipy pandas scikit-learn matplotlib)
 
# Step 2: installed through pip inside the env, in dependency order
PIP_PACKAGES=(gower plotly dash anthropic)
 
if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found in PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi
 
# 1. Create the environment (skip if it already exists)
if conda env list | awk '!/^#/ {print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists, skipping creation."
else
    echo ">>> Creating environment '$ENV_NAME' (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip
fi
 
# 2. Install the scientific stack with conda, one at a time in order
for pkg in "${CONDA_PACKAGES[@]}"; do
    echo ">>> conda install $pkg"
    conda install -y -n "$ENV_NAME" -c conda-forge "$pkg"
done
 
# 3. Install the rest with pip inside the environment, in order
for pkg in "${PIP_PACKAGES[@]}"; do
    echo ">>> pip install $pkg"
    conda run -n "$ENV_NAME" --no-capture-output python -m pip install "$pkg"
done
 
# 4. Verify that everything imports
echo ">>> Verifying installation"
conda run -n "$ENV_NAME" --no-capture-output python -c \
    "import pandas, anthropic, gower, sklearn, plotly, matplotlib, dash; print('All libraries imported successfully')"
 
echo
echo "Done. Activate the environment with:  conda activate $ENV_NAME"