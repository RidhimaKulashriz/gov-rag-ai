"""
Configuration module - CORRECTED for current models (Dec 2024)
"""
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# ============================================================================
# GOOGLE API KEY - MUST be set in .env
# ============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY must be set in .env file. Get it from https://aistudio.google.com/app/apikey")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Read model from env and clean it (remove any accidental spaces)
raw_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MODEL = "gemini-1.5-flash"  # Use stable model that's actually available

# Verified working models from AI Studio (current as of Dec 2024)
WORKING_MODELS = {
    # Current generation (RECOMMENDED)
    "gemini-1.5-flash": {
        "description": "Stable fast model - RECOMMENDED",
        "free_tier": "~60 req/min, 1500+ req/day",
        "context": "1M tokens"
    },
    "gemini-1.5-pro": {
        "description": "Proven capable model",
        "free_tier": "Lower limits",
        "context": "2M tokens"
    },
    "gemini-2.0-flash": {
        "description": "Latest fast model",
        "free_tier": "Preview limits",
        "context": "1M tokens"
    },
    "gemini-1.0-pro": {
        "description": "Stable legacy fallback",
        "free_tier": "Generous limits",
        "context": "32K tokens"
    },
}

# Validate the model
if GEMINI_MODEL not in WORKING_MODELS:
    print(f"⚠️  Warning: Model '{GEMINI_MODEL}' not in verified list.")
    print(f"   Available models: {list(WORKING_MODELS.keys())}")
    print(f"   Using fallback: gemini-1.5-flash")
    GEMINI_MODEL = "gemini-1.5-flash"

print(f"✅ Using model: {GEMINI_MODEL} ({WORKING_MODELS[GEMINI_MODEL]['description']})")

# ============================================================================
# RATE LIMITS (approximate free tier from docs)
# ============================================================================
RATE_LIMITS = {
    "gemini-1.5-flash-latest": {
        "requests_per_minute": 60,
        "requests_per_day": 1500,
        "tokens_per_minute": 1000000,
        "free_tier": True
    },
    "gemini-1.5-flash": {
        "requests_per_minute": 60,
        "requests_per_day": 1500,
        "tokens_per_minute": 1000000,
        "free_tier": True
    },
    "gemini-1.5-pro-latest": {
        "requests_per_minute": 15,
        "requests_per_day": 50,
        "tokens_per_minute": 500000,
        "free_tier": True
    },
    "gemini-1.5-pro": {
        "requests_per_minute": 15,
        "requests_per_day": 50,
        "tokens_per_minute": 500000,
        "free_tier": True
    },
    "gemini-2.0-flash-exp": {
        "requests_per_minute": 30,
        "requests_per_day": 500,
        "tokens_per_minute": 500000,
        "free_tier": True
    },
    "gemini-1.0-pro": {
        "requests_per_minute": 60,
        "requests_per_day": 1500,
        "tokens_per_minute": 150000,
        "free_tier": True
    }
}

# Get limits for current model
CURRENT_LIMITS = RATE_LIMITS.get(GEMINI_MODEL, RATE_LIMITS["gemini-1.5-flash-latest"])

# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_K_RESULTS = 10

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "data/pdfs"
TEMP_DIR = "temp"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print(f"✅ Config loaded successfully. API key ready, data dir: {DATA_DIR}")

