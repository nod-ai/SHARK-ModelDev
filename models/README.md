# Turbine-Models setup (source)

For private/gated models, make sure you have run `huggingface-cli login`.

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

