# SHARK torchbench exports and benchmarks

## Overview

This directory serves as a place for scripts and utilities to run a suite of benchmarked inference tasks, showing functionality and performance parity between SHARK/IREE and native torch.compile workflows. It is currently under development and benchmark numbers should not be treated as the best possible result with the current state of IREE compiler optimizations.

Eventually, we want this process to be a plug-in to the upstream torchbench process, and this will be accomplished by exposing the IREE methodology shown here as a compile/runtime backend for the torch benchmark classes. For now, it is set up for developers as a way to get preliminary results and achieve blanket functionality for the models listed in export.py.

The setup instructions provided here, in a few cases, use "gfx942" as the IREE/LLVM hip target. This is for MI300x accelerators -- you can find a mapping of AMD targets to their LLVM target architecture [here](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-architecture-table), and replace "gfx942" in the following documentation with your desired target.

## Setup (docker)

Use the dockerfile provided with the following build/run commands to execute in docker.
These commands assume a few things about your machine/distro, so please read them and make sure they do what you want.

```shell
docker build --platform linux/amd64 --tag shark_torchbench --file shark_torchbench.dockerfile .
```
```shell
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ./shark_torchbench_outputs:/SHARK-Turbine/models/turbine_models/custom_models/torchbench/outputs -w /SHARK-Turbine/models/turbine_models/custom_models/torchbench shark_torchbench:latest
```
```shell
python3 ./export.py --target=gfx942 --device=rocm --compile_to=vmfb --performance --inference --precision=fp16 --float16 --external_weights=safetensors --external_weights_dir=./torchbench_weights/ --output_csv=./outputs/torchbench_results_SHARK.csv
```


## Setup (source)

### Setup source code and prerequisites

 - pip install torch+rocm packages:
```shell
pip install torch==2.5.0.dev20240801+rocm6.1 torchvision==0.20.0.dev20240801+rocm6.1 torchaudio==2.4.0.dev20240801+rocm6.1 --index-url https://download.pytorch.org/whl/nightly/rocm6.1

```
 - Workaround amdsmi error in pre-release pytorch+rocm:
```shell
sudo apt install amd-smi-lib
sudo chown -R $USER:$USER /opt/rocm/share/amd_smi
python3 -m pip install /opt/rocm/share/amd_smi
```
 - Clone torch and expose benchmarking code as a relative module:
```shell
git clone https://github.com/pytorch/pytorch
cd pytorch/benchmarks
touch __init__.py
cd ../..
```
 - Clone and install pytorch benchmark modules:
```shell
git clone https://github.com/pytorch/benchmark
cd benchmark
python3 install.py --models BERT_pytorch Background_Matting LearningToPaint alexnet dcgan densenet121 hf_Albert hf_Bart hf_Bert hf_GPT2 hf_T5 mnasnet1_0 mobilenet_v2 mobilenet_v3_large nvidia_deeprecommender pytorch_unet resnet18 resnet50 resnet50_32x4d shufflenet_v2_x1_0 squeezenet1_1 timm_nfnet timm_efficientnet timm_regnet timm_resnest timm_vision_transformer timm_vovnet vgg16
pip install -e .
cd ..
```

### Export and compile

```shell
python ./export.py --target=gfx942 --device=rocm --compile_to=vmfb --performance --inference --precision=fp16 --float16 --external_weights=safetensors --external_weights_dir=./torchbench_weights/
```

### Example of manual benchmark using export and IREE runtime CLI (mobilenet_v3_large)

```shell
 python ./export.py --target=gfx942 --device=rocm --compile_to=vmfb --performance --inference --precision=fp16 --float16 --external_weights=safetensors --external_weights_dir=./torchbench_weights/ --model_id=mobilenet_v3_large

iree-benchmark-module --module=generated/mobilenet_v3_large_256_fp16_gfx942.vmfb --input=@generated/mobilenet_v3_large_input0.npy --parameters=model=./torchbench_weights/mobilenet_v3_large_fp16.irpa --device=hip://0 --device_allocator=caching --function=main --benchmark_repetitions=10
```