#!/usr/bin/env bash
set -e
PYTHON=/home/mohamed-ashraf/Desktop/projects/env/bin/python
if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi
export MPLCONFIGDIR=/tmp/matplotlib
if [ "$#" -eq 0 ]; then
  set -- --imgs-dir /home/mohamed-ashraf/Desktop/projects/sat-project/data/test/samples_prepared/imgs --model-type dl --output-dir predictions
fi
"$PYTHON" test.py "$@"
