#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv311/bin/activate
python apple_face_states_v3.py
