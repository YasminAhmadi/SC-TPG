```bash
#!/bin/bash
#SBATCH --job-name=train
##SBATCH --account=your_allocation_name
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REAL_HOME="$HOME"

echo "[INFO] Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Start time: $(date)"


# 0) Create a job-local temporary directory
TMPROOT="${SLURM_TMPDIR:-/local/scratch/${USER}.${SLURM_JOB_ID:-manual}}"
mkdir -p "$TMPROOT"
echo "[INFO] TMPROOT=$TMPROOT"

# Isolate HOME and XDG directories so that multiple jobs do not write to the
# real home directory or interfere with each other.
export HOME="$TMPROOT/home"
mkdir -p "$HOME"

export XDG_CACHE_HOME="$HOME/.cache"
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_DATA_HOME="$HOME/.local/share"

mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"


# 1) Locate the shared StarCraft II installation
# Set SC2_MASTER to the shared StarCraft II installation directory.
# Example:
#   export SC2_MASTER=/path/to/StarCraftII
SC2_MASTER="${SC2_MASTER:-/path/to/StarCraftII}"

if [ ! -d "$SC2_MASTER/Versions" ]; then
  echo "[FATAL] Invalid SC2_MASTER: $SC2_MASTER"
  echo "[FATAL] Expected to find: $SC2_MASTER/Versions"
  exit 2
fi

echo "[INFO] SC2_MASTER=$SC2_MASTER"


# 2) Copy StarCraft II to job-local storage
# Running SC2 directly from shared storage can cause write conflicts or I/O
# bottlenecks when multiple jobs run at the same time. Therefore, each job uses
# its own local copy.
export SC2PATH="$TMPROOT/StarCraftII"
export SC2_PATH="$SC2PATH"

mkdir -p "$SC2PATH"
echo "[INFO] Copying StarCraft II to local storage: $SC2PATH"

# Replays are excluded to reduce I/O.
rsync -a --delete --exclude='Replays/**' "$SC2_MASTER/" "$SC2PATH/"

# Create a local replay directory for this job.
mkdir -p "$SC2PATH/Replays"

# Some environments expect either "Maps" or "maps".
if [ -d "$SC2PATH/Maps" ]; then
  ln -sfn "$SC2PATH/Maps" "$SC2PATH/maps" || true
fi

# Ensure the SC2 binary is executable.
chmod +x "$SC2PATH"/Versions/*/SC2_x64 2>/dev/null || true

SC2_BIN="$(ls -1 "$SC2PATH"/Versions/*/SC2_x64 2>/dev/null | head -n 1 || true)"

if [ -z "$SC2_BIN" ] || [ ! -x "$SC2_BIN" ]; then
  echo "[FATAL] SC2_x64 was not found or is not executable under:"
  echo "[FATAL] $SC2PATH/Versions"
  ls -lah "$SC2PATH/Versions" || true
  exit 3
fi

echo "[INFO] SC2_BIN=$SC2_BIN"


# 3) Limit the number of SC2 instances per node
# SC2 can be resource-heavy. This lock-based slot mechanism prevents too many
# SC2 instances from running on the same node at once.
SC2_MAX_PER_NODE="${SC2_MAX_PER_NODE:-2}"

SLOT_DIR="/tmp/sc2_slots_${USER}"
SLOT_LOCK="$SLOT_DIR/lock"

mkdir -p "$SLOT_DIR"

slot=""

while [ -z "$slot" ]; do
  exec 9>"$SLOT_LOCK"
  flock 9

  for i in $(seq 0 $((SC2_MAX_PER_NODE - 1))); do
    if [ ! -e "$SLOT_DIR/slot.$i" ]; then
      slot="$i"
      echo "${SLURM_JOB_ID:-manual}" > "$SLOT_DIR/slot.$i"
      break
    fi
  done

  flock -u 9
  exec 9>&-

  if [ -z "$slot" ]; then
    echo "[INFO] SC2 slots are full on this node."
    echo "[INFO] Maximum allowed per node: $SC2_MAX_PER_NODE"
    echo "[INFO] Waiting 10 seconds..."
    sleep 10
  fi
done

cleanup() {
  rm -f "$SLOT_DIR/slot.$slot" 2>/dev/null || true
}

trap cleanup EXIT

echo "[INFO] Acquired SC2 slot: $slot"
echo "[INFO] Maximum SC2 instances per node: $SC2_MAX_PER_NODE"


# 4) Load Python environment
# Adjust the module and virtual environment paths for your cluster.
module load python/3.11

# Set VENV_DIR before submission if your virtual environment is elsewhere.
# Example:
#   export VENV_DIR=/path/to/venv
VENV_DIR="${VENV_DIR:-$REAL_HOME/venvs/tpg-sc2}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[FATAL] Python virtual environment not found:"
  echo "[FATAL] $VENV_DIR"
  exit 4
fi

source "$VENV_DIR/bin/activate"

# Set PROJECT_DIR before submission if the project is elsewhere.
# Example:
#   export PROJECT_DIR=/path/to/your/project
PROJECT_DIR="${PROJECT_DIR:-$REAL_HOME/projects/your_project}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "[FATAL] Project directory not found:"
  echo "[FATAL] $PROJECT_DIR"
  exit 5
fi

cd "$PROJECT_DIR"


# 5) Configure CPU threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[INFO] SC2PATH=$SC2PATH"
echo "[INFO] HOME=$HOME"
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] VENV_DIR=$VENV_DIR"
echo "[INFO] OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo "[INFO] TASK_NAME=${TASK_NAME:-<unset>}"
echo "[INFO] POLICY_NAME=${POLICY_NAME:-<unset>}"
echo "[INFO] SEED=${SEED:-<unset>}"


# 6) Run training
python -u train.py
```
