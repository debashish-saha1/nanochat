# Comment out rl part in thr report.py
# "chat-rl.md",
# "chat-evaluation-rl.md"
# """ --- IGNORE ---
# # the expected files and their order
# EXPECTED_FILES = [
#     "tokenizer-training.md",
#     "tokenizer-evaluation.md",
#     "base-model-training.md",
#     "base-model-loss.md",
#     "base-model-evaluation.md",
#     "midtraining.md",
#     "chat-evaluation-mid.md",
#     "chat-sft.md",
#     "chat-evaluation-sft.md",
#     # "chat-rl.md",
#     # "chat-evaluation-rl.md",
# ] --- IGNORE ---


#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# Disable SSL verification for corporate networks (Mastercard, etc.)
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_CERT_FILE=""
mkdir -p $NANOCHAT_BASE_DIR

# Initialize the report directory structure
echo "Initializing report..."
python -m nanochat.report reset

# download dataset if you don't have it already
echo "Downloading dataset shards..."
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

# Download identity conversations for midtraining
echo "Downloading identity conversations..."
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
echo "Training tokenizer on ~2B characters..."
python -m scripts.tok_train --max-chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
echo "Evaluating tokenizer..."
python -m scripts.tok_eval

# Wait for dataset download to complete
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Define model tag
MODEL_TAG="speedrun_fast"

# base training
echo "Starting base training..."
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=1024 --eval-every=50 --eval-tokens=4096 --core-metric-every=50 --core-metric-max-per-task=12 --sample-every=50 --num-iterations=500 --model-tag=$MODEL_TAG --device-type=cpu
python -m scripts.base_loss --device-batch-size=1 --split-tokens=4096 --device-type=cpu
python -m scripts.base_eval --max-per-task=16 --device-type=cpu

# midtraining
echo "Starting midtraining..."
# increase iterations for better midtraining = 2*base iterations
python -m scripts.mid_train --max-seq-len=512 --device-batch-size=1 --eval-every=50 --eval-tokens=4096 --total-batch-size=1024 --num-iterations=500 --device-type=cpu

# eval
echo "Evaluating midtrained model..."
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20 --device-type=cpu 

# SFT
echo "Starting supervised finetuning..."
python -m scripts.chat_sft --device-batch-size=1 --target-examples-per-step=4 --num-iterations=500 --eval-steps=4 --eval-metrics-max-problems=16 --device-type=cpu 

echo "Starting sft evaluation..."
python -m scripts.chat_eval -i sft --max-new-tokens=128 --max-problems=16 --device-type=cpu
echo "Supervised finetuning complete."

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
python -m scripts.chat_web

# Generate final report
echo "Generating report..."
python -m nanochat.report generate

# Chat with the model
python -m scripts.chat_cli -p "Hello, who won the world series in 2020?" --device-type=cpu
