
# --- Layer 1: Services-only companies (JD: "explicitly do not want") ---
PURE_SERVICES_COMPANIES = {
    "tcs", "tata consultancy services",
    "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl",
}

# --- Layer 1: CV/Speech/Robotics — wrong domain (JD: "re-learning fundamentals") ---
WRONG_DOMAIN_KEYWORDS = {
    "computer vision", "object detection", "image segmentation",
    "speech recognition", "asr", "text to speech", "tts",
    "robotics", "ros", "slam", "autonomous driving",
    "action recognition", "pose estimation",
}

# --- Layer 1: Must-have skills from JD ---
# Candidate must match AT LEAST 1 to survive Layer 1
MUST_HAVE_SKILLS = {
    # Embeddings / retrieval
    "embeddings", "sentence transformers", "sentence-transformers",
    "bge", "e5", "openai embeddings", "dense retrieval",
    # Vector DBs / hybrid search
    "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss",
    # Ranking / evaluation
    "ranking", "retrieval", "ndcg", "mrr", "map",
    "learning to rank", "ltr", "bm25", "hybrid search",
    # Core language
    "python",
}

# --- Layer 1: Recent-only LLM wrapper detection ---
# If skills are ONLY from this set with no pre-LLM signal → weak flag
LLM_WRAPPER_ONLY_SKILLS = {
    "langchain", "llamaindex", "openai", "chatgpt",
    "gpt-4", "gpt4", "claude", "gemini",
}

# --- Layer 1: NLP/IR presence check (needed to override wrong-domain flag) ---
NLP_IR_KEYWORDS = {
    "nlp", "natural language processing", "information retrieval",
    "text classification", "named entity recognition", "ner",
    "question answering", "semantic search", "ranking",
    "retrieval", "search", "recommendation",
}

# --- Thresholds ---
MIN_YEARS_EXPERIENCE = 3          # JD says 5-9; under 3 is definitive no
MAX_INACTIVE_MONTHS = 18          # JD: "hasn't written production code in 18 months"
