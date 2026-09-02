#!/bin/bash

# --- Core Configuration ---
DATA_PATH=$1                                                                  # Index file of data which need to be eval
OUTPUT_DIR=$2                                                                 # Output csv path
API_KEY=$3                                                                    # API key for LLM
API_URL=$4                                                                    # Chat completions API URL for LLM
OCR_DET_MODEL_DIR=$5                                                          # Path to PaddleOCR v5 detection model: PP-OCRv5_server_det
OCR_REC_MODEL_DIR=$6                                                          # Path to PaddleOCR v5 recognition model:PP-OCRv5_server_rec
EVAL_TYPE=${7:-"None"}                                                        # eval type: instruction_following, readability, ocr, ocr_vl, aes, text_position_vlm, math_vlm, None
N_PROCESS=1                                                                   # Number of worker threads; higher values may cause more failures.
MODEL_NAME=Qwen3.5-397B-A17B                                                  # which LLM to be used in VLM-based evaluation
DEVICE=cuda:0                                                                 # Specify the GPU ID to use
API_HANDLER_TYPE=VLMAPIHandler                                                # API handler type
AESTHETIC_METHOD=${AESTHETIC_METHOD:-"vlm"}                                   # vlm or aes; aes uses local paths configured in code

OUTPUT_CSV_PATH="${OUTPUT_DIR%/}/eval_result.csv"                             # Default output csv path

VISUAL_NAME=${VISUAL_NAME:-"current"}

# Execute the python script with long-form options for readability
python run.py \
  --data-path "$DATA_PATH" \
  --output-path "$OUTPUT_CSV_PATH" \
  --api-key "$API_KEY" \
  --api-url "$API_URL" \
  --n-process $N_PROCESS \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --aesthetic-method "$AESTHETIC_METHOD" \
  --ocr-det-model-dir "$OCR_DET_MODEL_DIR" \
  --ocr-rec-model-dir "$OCR_REC_MODEL_DIR" \
  --api-handler-type "$API_HANDLER_TYPE" \
  --eval_type "$EVAL_TYPE"

python utils/visualization.py \
  --csv-paths "$OUTPUT_CSV_PATH" \
  --names "$VISUAL_NAME" \
  --save-dir "$OUTPUT_DIR"
