"""
Configuration settings for Multi-Agent RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
SESSIONS_DIR = DATA_DIR / "sessions"

# Create directories if they don't exist
for dir_path in [DATA_DIR, DOCUMENTS_DIR, VECTOR_STORE_DIR, SESSIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")
EXA_API_KEY = os.getenv("EXA_API_KEY", "71b5d560-e85e-480e-a2ca-f83139512385")

# Model Configuration
MODEL_CONFIG = {
    "primary_model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu",
    "torch_dtype": "float16",  # Use float16 for efficiency
    "use_quantization": os.getenv("USE_QUANTIZATION", "false").lower() == "true",
}

# Generation Parameters
GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,  # Number of chunks to retrieve
    "similarity_threshold": 0.5,
}

# Memory Configuration
MEMORY_CONFIG = {
    "max_message_buffer": 10,  # Keep last 10 messages
    "session_ttl": 86400 * 7,  # 7 days in seconds
    "enable_summarization": True,
    "summarization_trigger": 10,  # Summarize after 10 messages
}

# Search Configuration
SEARCH_CONFIG = {
    "default_num_results": 5,
    "enable_highlights": True,
    "highlights_per_url": 3,
    "fallback_to_searxng": True,
}

# Citation Configuration
CITATION_CONFIG = {
    "min_confidence": 0.5,
    "context_window": 100,  # Characters before/after quote
    "style": "inline",  # inline or detailed
}

# Agent Configuration
AGENT_CONFIG = {
    "enable_parallel": False,  # Parallel agent execution (for future)
    "timeout": 60,  # Agent timeout in seconds
    "retry_attempts": 3,
}

# Vercel Configuration (for deployment)
VERCEL_CONFIG = {
    "max_duration": 60,
    "memory": 3008,
    "region": "iad1",
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# System Prompts
SYSTEM_PROMPTS = {
    "master_agent": """You are an intelligent AI assistant with access to multiple specialized agents. 
Your role is to:
1. Analyze user queries carefully
2. Coordinate with specialized agents (search, retrieval, summarization)
3. Synthesize information from multiple sources
4. Provide accurate answers with proper citations
5. Maintain conversation context and memory

Always cite your sources and be transparent about the information you use.""",

    "search_agent": """You are a search specialist. Your role is to:
1. Formulate effective search queries
2. Retrieve relevant web content
3. Extract key information from search results
4. Identify the most relevant sources""",

    "summarization_agent": """You are a summarization expert. Your role is to:
1. Condense long documents into concise summaries
2. Preserve key information and facts
3. Maintain context and coherence
4. Highlight important points""",

    "citation_agent": """You are a citation specialist. Your role is to:
1. Track all sources of information
2. Extract exact quotes from documents
3. Map claims to specific sources
4. Provide accurate citations with locations""",
}

# Prompts for specific tasks
TASK_PROMPTS = {
    "extract_text_from_image": "Extract all text from this image. Provide the text exactly as it appears, preserving formatting where possible.",
    
    "answer_with_context": """Based on the provided context, answer the following question. 
Use ONLY information from the context. If the answer is not in the context, say so.
Cite specific sources for all claims using [Source N] format.

Context:
{context}

Question: {question}

Answer:""",

    "summarize_document": """Summarize the following document concisely. 
Focus on the main points and key findings.

Document:
{document}

Summary:""",

    "extract_claims": """Extract all factual claims from the following text.
List each claim separately.

Text:
{text}

Claims:""",
}

# Error messages
ERROR_MESSAGES = {
    "no_context": "I couldn't find relevant information in the available documents to answer your question.",
    "search_failed": "The search service is temporarily unavailable. Please try again.",
    "model_error": "An error occurred while processing your request. Please try again.",
    "citation_error": "Could not extract citations for this response.",
    "session_error": "Could not load or save session data.",
}

# Success messages
SUCCESS_MESSAGES = {
    "document_uploaded": "Document successfully processed and indexed.",
    "session_created": "New session created successfully.",
    "search_complete": "Search completed. Found {num_results} results.",
}
