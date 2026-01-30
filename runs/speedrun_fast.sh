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
# Disable SSL verification for corporate networks (to avoid certificate issues)
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_CERT_FILE=""
mkdir -p $NANOCHAT_BASE_DIR

# # download dataset if you don't have it already
# python -m nanochat.dataset -n 8
# python -m nanochat.dataset -n 240 &
# DATASET_DOWNLOAD_PID=$!

# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
echo "Training tokenizer on ~2B characters..."
python -m scripts.tok_train --max-chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
echo "Evaluating tokenizer..."
python -m scripts.tok_eval


# base training
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=1024 --eval-every=50 --eval-tokens=4096 --core-metric-every=50 --core-metric-max-per-task=12 --sample-every=50 --num-iterations=5000 --model-tag=$MODEL_TAG
python -m scripts.base_loss --device-batch-size=1 --split-tokens=4096
python -m scripts.base_eval --max-per-task=16

# midtraining
echo "Starting midtraining..."
# increase iterations for better midtraining = 2*base iterations
python -m scripts.mid_train --max-seq-len=512 --device-batch-size=1 --eval-every=50 --eval-tokens=4096 --total-batch-size=1024 --num-iterations=5000 --device-type=cpu

# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20 

# SFT
echo "Starting supervised finetuning..."
python -m scripts.chat_sft --device-batch-size=1 --target-examples-per-step=4 --num-iterations=5000 --eval-steps=4 --eval-metrics-max-problems=16 --device-type=cpu 

echo "..........sft training complete. Starting sft evaluation..."
python -m scripts.chat_eval -i sft --max-new-tokens=128 --max-problems=16
echo "..........Supervised finetuning training and evaluation complete."

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate

python -m scripts.chat_cli -p "Hello, who won the world series in 2020?"
