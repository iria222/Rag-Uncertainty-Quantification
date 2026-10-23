"""
This module contains configuration settings for the RAG pipeline.
"""

from pathlib import Path
import os
import torch

APPLY_SELF_EVALUATION = True
DEBUG_MODE = True
LOAD_ON_DEMAND = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

QUANTIZATION = "8bit"

MAX_WORKERS = 4
WORKERS = min(MAX_WORKERS, os.cpu_count() or 1)

#################################
# DIRECTORY CONFIGURATION
#################################

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = DATA_DIR / 'cache'
MANIFEST_DIR = CACHE_DIR / 'manifests.json'
CHUNK_DIR = CACHE_DIR / 'chunks.json'
INDEX_DIR = CACHE_DIR / 'index.faiss'
