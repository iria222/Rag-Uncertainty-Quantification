"""
This module contains configuration settings for the RAG pipeline.
"""

from pathlib import Path
import os
import torch

APPLY_SELF_EVALUATION = True
SELF_EVAL_THRESHOLD = None


LOAD_ON_DEMAND = True # It is highly recommended to keep this option as True if your vram is limited (8gb or less)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_SAVE_FREQUENCY = 10

#################################
# DEBUGGING INFORMATION
#################################

DEBUG_MODE = True
SHOW_METRICS = True

#################################
# CHUNKING CONFIGURATION
#################################

CACHE_VERSION = 1
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

#################################
# RETRIEVAL CONFIGURATION
#################################

TOP_K_RETRIEVED = 100
TOP_K_RERANKED = 5

#################################
# MODEL CONFIGURATION
#################################

EMB_MODEL_NAME = 'google/embeddinggemma-300m'
RERANK_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
GEN_MODEL_NAME = 'google/gemma-4-E2B-it'

ALLOWED_QUANTIZATIONS = {"4bit", "8bit", None}
# Quantization for generation model
# Available options: 4bit and 8bit
# Set QUANTIZATION = None if you don't want to use a quantized version of the model 
QUANTIZATION = "8bit"

MAX_WORKERS = 4
WORKERS = min(MAX_WORKERS, os.cpu_count() or 1)

#################################
# MODEL CONFIGURATION - LLM-AS-JUDGE
#################################

JUDGE_MODEL_NAME = "Qwen/Qwen3-8B"
JUDGE_QUANTIZATION = "4bit"
JUDGE_MAX_TOKENS = 512

#################################
# DIRECTORY CONFIGURATION
#################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
DATA_DIR = DATA_ROOT / 'corpus'
OUTPUT_DIR = PROJECT_ROOT / 'output'
SELF_EVAL_FEWSHOT_PATH = DATA_ROOT / 'self_eval' / 'selfeval_fewshot.json'
CACHE_DIR = DATA_ROOT / 'cache'
MANIFEST_DIR = CACHE_DIR / 'manifests.json'
CHUNK_DIR = CACHE_DIR / 'chunks.json'
INDEX_DIR = CACHE_DIR / 'index.faiss'
