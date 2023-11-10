# LLAMA 2 Inference

This example require some extra dependencies. Here's an easy way to get it running on a fresh server. You probably won't need to edit this script except to log into huggingface.

```bash
#!/bin/bash

# Step 1. Get a server to run this. Xida recommends [tensordock](https://tensordock.com/) for less than $2 an hour.

# Step 2: Install Git
sudo apt install -y git

# Step 3: Clone SHARK-Turbine repository
git clone https://github.com/nod-ai/SHARK-Turbine.git

# Step 4: Change directory
cd SHARK-Turbine

# Step 5: Install requirements from requirements.txt
pip install -r requirements.txt

# Step 6: Install additional Python packages
pip install -y transformers sentencepiece protobuf

# or use huggingface-cli login --token (your token) for headless
huggingface-cli login

# Step 7: Run the Python script
python examples/llama2_inference/stateless_llama.py

```
