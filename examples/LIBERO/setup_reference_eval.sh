#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_REPO="$PROJECT_ROOT/external_dependencies/LIBERO"
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
SIM_ENV="$PROJECT_ROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv"
cd "$PROJECT_ROOT"
if [[ ! -d "$LIBERO_REPO/.git" ]]; then
  mkdir -p "$LIBERO_REPO"
  git -C "$LIBERO_REPO" init --quiet
  git -C "$LIBERO_REPO" remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git
  git -C "$LIBERO_REPO" fetch --depth 1 origin "$LIBERO_COMMIT"
  git -C "$LIBERO_REPO" checkout --detach FETCH_HEAD
elif [[ "$(git -C "$LIBERO_REPO" rev-parse HEAD)" != "$LIBERO_COMMIT" ]]; then
  echo "LIBERO must be at $LIBERO_COMMIT: $LIBERO_REPO" >&2
  exit 1
fi
if [[ ! -x "$SIM_ENV/bin/python" ]]; then
  uv venv "$SIM_ENV" --python "$PROJECT_ROOT/.venv/bin/python"
fi
# Reuse the existing model environment's PyTorch and utility packages. The sim
# environment supplies its own pinned MuJoCo/robosuite/NumPy dependencies.
SIM_ENV="$SIM_ENV" "$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import sysconfig
root = Path.cwd()
site = Path(os.environ['SIM_ENV']) / 'lib/python3.12/site-packages'
(site / 'minimal_groot_shared.pth').write_text(sysconfig.get_path('purelib') + '\n' + str(root) + '\n')
PY
uv pip install --python "$SIM_ENV/bin/python" \
  robosuite==1.4.0 bddl==1.0.1 gym==0.26.2 easydict==1.13 future==1.0.0 \
  h5py==3.14.0 mujoco==3.3.1 numpy==1.26.4 numba==0.65.1 llvmlite==0.47.0 \
  opencv-python==4.10.0.84 websockets==15.0.1 imageio==2.37.0 pyyaml==6.0.3 termcolor==3.2.0
uv pip install --python "$SIM_ENV/bin/python" --no-deps -e "$LIBERO_REPO" \
  --config-settings editable_mode=compat
if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
  printf 'n\n' | "$SIM_ENV/bin/python" -c 'from libero.libero import get_libero_path; print(get_libero_path("bddl_files"))'
fi
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$SIM_ENV/bin/python" - <<'PY'
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                        camera_heights=256, camera_widths=256)
try:
    env.seed(7)
    env.reset()
    env.set_init_state(suite.get_task_init_states(0)[0])
    env.step([0.0] * 6 + [-1.0])
    print('LIBERO Spatial simulator smoke passed')
finally:
    env.close()
PY
