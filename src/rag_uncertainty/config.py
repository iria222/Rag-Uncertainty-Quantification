"""
This module contains configuration settings for the RAG pipeline.
"""

from pathlib import Path
import os
import torch

APPLY_SELF_EVALUATION = True
SELF_EVAL_THRESHOLD = None

DEBUG_MODE = True
LOAD_ON_DEMAND = True # It is highly recommended to keep this option as True if your vram is limited (8gb or less)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_SAVE_FREQUENCY = 10

#################################
# CHUNKING CONFIGURATION
#################################

CACHE_VERSION = 1
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

#################################
# MODEL CONFIGURATION
#################################

EMB_MODEL_NAME = 'google/embeddinggemma-300m'
RERANK_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
GEN_MODEL_NAME = 'google/gemma-4-E2B-it'

# Available options: 4bit and 8bit
# Set QUANTIZATION = None if you don't want to use a quantized version of the model 
QUANTIZATION = "8bit"

MAX_WORKERS = 4
WORKERS = min(MAX_WORKERS, os.cpu_count() or 1)

#################################
# DIRECTORY CONFIGURATION
#################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
SELF_EVAL_FEWSHOT_PATH = DATA_DIR / 'self_eval' / 'selfeval_fewshot.json'
CACHE_DIR = DATA_DIR / 'cache'
MANIFEST_DIR = CACHE_DIR / 'manifests.json'
CHUNK_DIR = CACHE_DIR / 'chunks.json'
INDEX_DIR = CACHE_DIR / 'index.faiss'
