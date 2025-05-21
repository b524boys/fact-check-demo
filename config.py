# System configuration

# Model paths
QWEN_MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
DEEPSEEK_MODEL_PATH = "/home/featurize/data/models/deepseek_r1"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# Google search API settings
GOOGLE_API_KEY = "AIzaSyChUp5uzx3AMOlqeBcmwIk5moL0bd6uclo"
GOOGLE_SEARCH_ENGINE_ID = "96c34a424e2184b2a"

# RAG settings
VECTOR_DIMENSION = 768
INDEX_PATH = "data/cache/faiss_index"
BATCH_SIZE = 32
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5

# Device settings
DEVICE = "cuda"

# Cache settings
CACHE_DIR = "data/cache"
MODEL_CACHE_DIR = "/home/featurize/data/models"
ENABLE_CACHE = True