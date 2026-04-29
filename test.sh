#!/usr/bin/env bash
set -e
PYTHON=/home/mohamed-ashraf/Desktop/projects/env/bin/python
if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi
if [ "$#" -eq 0 ]; then
  set -- --imgs-dir /home/mohamed-ashraf/Desktop/projects/sat-project/data/test/samples_prepared/imgs --masks-dir /home/mohamed-ashraf/Desktop/projects/sat-project/data/test/samples_prepared/masks --model-type dl --output predictions.txt
fi
"$PYTHON" test.py "$@"
