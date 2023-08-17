#!/bin/bash

git clone https://github.com/jansel/pytorch-jit-paritybench.git

cd pytorch-jit-paritybench

git checkout 7e55a422588c1d1e00f35a3d3a3ff896cce59e18
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install expecttest

cd ..

pip install ./compiler
pip install ./runtime
pip install . --no-deps
source build/iree/.env

python python/test/generated/main.py --tests-dir ./pytorch-jit-paritybench --limit 100 -j 4 --no-log


