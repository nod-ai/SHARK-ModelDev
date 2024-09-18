# SHARK torchbench exports and benchmarks

### Setup

 - pip install torch+rocm packages:
```shell
pip install --pre torch==2.5.0.dev20240801+rocm6.1 torchvision==0.20.0.dev20240801+rocm6.1 torchaudio==2.4.0.dev20240801%2Brocm6.1 --index-url https://download.pytorch.org/whl/nightly/rocm6.1

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