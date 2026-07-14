#!/bin/bash
#SBATCH --job-name=project_cvcs
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/work/cvcs2026/pixel_pioneers/out/jobs/%j.out
#SBATCH --account=cvcs2026

PROJ_DIR="/work/cvcs2026/pixel_pioneers"
VENV_DIR="$PROJ_DIR/.venv"

# module load python/3.11.11-gcc-11.4.0

# if [ ! -f "$VENV_DIR/bin/activate" ]; then
#    echo "[job] creating venv..."
#    python -m venv "$VENV_DIR"
# fi

source "$VENV_DIR/bin/activate"

# pip install --upgrade pip
# pip install -r "$PROJ_DIR/requirements.txt"

python "$PROJ_DIR/src/py/metrics_main.py"
