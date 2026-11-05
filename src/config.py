"""Shared project configuration"""
from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / 'EnterpriseRAG-Bench'
OUTPUT_ROOT = PROJECT_ROOT / 'data'
MANIFEST_ROOT = PROJECT_ROOT / 'manifests'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 0
BGE_MODEL = 'BAAI/bge-base-en-v1.5'
BGE_REVISION = 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
DEV_COUNTS = {'single-fact': 4, 'multi-hop': 4, 'corpus-global': 0, 'fuzzy': 4}

# type of question and appropriate question_type from provided in the dataset
QUESTION_CATEGORIES = {
    'single-fact' : ['basic'],
    'multi-hop' : ['intra_document_reasoning', 'project_related'],
    'corpus-global' : ['completeness', 'high_level', 'conflicting_info'],
    'fuzzy': ['semantic', 'constrained']
    }

TOTAL_QUESTIONS = 200

QUESTIONS_BY_TYPES_CNT = {'basic':50,
                  'intra_document_reasoning':25,
                  'project_related':25,
                  'completeness': 20,
                  'high_level':10, 
                  'conflicting_info':20,
                  'semantic':30, 
                  'constrained':20
                  }
