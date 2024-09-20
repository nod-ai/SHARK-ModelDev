FROM rocm/dev-ubuntu-22.04:6.1.2

# ######################################################
# # Install MLPerf+Shark reference implementation
# ######################################################
ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN apt-get update && apt-get install -y \
ffmpeg libsm6 libxext6 git wget unzip \
  software-properties-common git \
  build-essential curl cmake ninja-build clang lld vim nano python3.10-dev python3.10-venv && \
  apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel && \
    pip install pybind11 'nanobind<2' numpy==1.* pandas && \
    pip install hip-python hip-python-as-cuda -i https://test.pypi.org/simple

# Rust requirements
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

SHELL ["/bin/bash", "-c"]

# Disable apt-key parse waring
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

######################################################
# Install SHARK-Turbine
######################################################
RUN pip3 install torch==2.4.0+rocm6.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
RUN pip3 install --pre iree-compiler==20240920.1022 iree-runtime==20240920.1022 -f https://iree.dev/pip-release-links.html

RUN apt install amd-smi-lib && sudo chown -R $USER:$USER /opt/rocm/share/amd_smi && python3 -m pip install /opt/rocm/share/amd_smi
# Install turbine-models, where the export is implemented.

ENV TB_SHARK_DIR=/SHARK-Turbine/models/turbine_models/custom_models/torchbench

RUN git clone https://github.com/nod-ai/SHARK-Turbine -b torchbench \
  && cd SHARK-Turbine \
  && pip install --pre --upgrade -e models -r models/requirements.txt \
  && cd $TB_SHARK_DIR \
  && git clone https://github.com/pytorch/pytorch \ 
  && cd pytorch/benchmarks \ 
  && touch __init__.py && cd ../.. \
  && git clone https://github.com/pytorch/benchmark && cd benchmark \
  && python3 install.py --models BERT_pytorch Background_Matting LearningToPaint alexnet dcgan densenet121 hf_Albert hf_Bart hf_Bert hf_GPT2 hf_T5 mnasnet1_0 mobilenet_v2 mobilenet_v3_large nvidia_deeprecommender pytorch_unet resnet18 resnet50 resnet50_32x4d shufflenet_v2_x1_0 squeezenet1_1 timm_nfnet timm_efficientnet timm_regnet timm_resnest timm_vision_transformer timm_vovnet vgg16 \
  && pip install -e .

ENV HF_HOME=/models/huggingface/

# initialization settings for CPX mode
ENV HSA_USE_SVM=0
ENV HSA_XNACK=0