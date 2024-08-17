# Turbine-Models setup (source)

For private/gated models, make sure you have run `huggingface-cli login`.

For MI Instinct:
```bash
#!/bin/bash
sudo apt install -y git

# Clone and build IREE at the shared/sdxl_quantized branch
git clone https://github.com/iree-org/iree && cd iree
git checkout shared/sdxl_quantized
git submodule update --init
python -m venv iree.venv
pip install pybind11 numpy nanobind
cmake -S . -B build-release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=`which clang` -DCMAKE_CXX_COMPILER=`which clang++` \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$(which python3)" && \
  cmake --build build-release/

export PYTHONPATH=/path/to/iree/build-release/compiler/bindings/python:/path/to/iree/build-release/runtime/bindings/python

# Clone and setup turbine-models
cd ..
git clone https://github.com/nod-ai/SHARK-Turbine.git && cd SHARK-Turbine
git checkout merge_punet_sdxl
pip install torch==2.5.0.dev20240801 torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r models/requirements.txt
pip uninstall -y iree-compiler iree-runtime

pip install -e models

# Run sdxl tests.
python models/turbine_models/tests/sdxl_test.py pytest --device=rocm --rt_device=hip --iree_target_triple=gfx942 --external_weights=safetensors --precision=fp16 --clip_spec=mfma --unet_spec=mfma --vae_spec=mfma

# Generate an image.
# To reuse test artifacts/weights, add: --pipeline_dir=./test_vmfbs --external_weights_dir=./test_weights
python models/turbine_models/custom_models/sd_inference/sd_pipeline.py --hf_model_name=stabilityai/stable-diffusion-xl-base-1.0 --device=hip://0 --precision=fp16 --external_weights=safetensors --iree_target_triple=gfx942 --vae_decomp_attn --clip_decomp_attn --use_i8_punet --width=1024 --height=1024 --num_inference_steps=20 --benchmark=all --verbose

```
For GFX11 (RDNA3 Discrete GPUs/Ryzen laptops) the following setup is validated:
```bash
#!/bin/bash

# clone and install dependencies
sudo apt install -y git
git clone https://github.com/nod-ai/SHARK-Turbine.git
cd SHARK-Turbine
pip install torch==2.5.0.dev20240801 torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r models/requirements.txt

# do an editable install from the cloned SHARK-Turbine
pip install --editable models

# Log in with Hugging Face CLI if token setup is required
if [[ $YOUR_HF_TOKEN == hf_* ]]; then
    huggingface login --token $YOUR_HF_TOKEN
    echo "Logged in with YOUR_HF_TOKEN."
elif [ -f ~/.cache/huggingface/token ]; then
    # Read token from the file
    TOKEN_CONTENT=$(cat ~/.cache/huggingface/token)
    
    # Check if the token starts with "hf_"
    if [[ $TOKEN_CONTENT == hf_* ]]; then
        echo "Already logged in with a Hugging Face token."
    else
        echo "Token in file does not start with 'hf_'. Please log into huggingface to download models."
        huggingface-cli login
    fi
else
    echo "Please log into huggingface to download models."
    huggingface-cli login
fi

