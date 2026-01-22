# config.py - UPDATED CONFIGURATION
"""
Configuration & Master Data for Invoice Extraction System
UPDATED: Better defaults and OCR settings
"""

# Tractor Model Master Data (from sample documents)
# ADD YOUR MODELS HERE
TRACTOR_MODELS = {
    "MF-1035 DI": {"hp": 35, "category": "MF"},
    "MF-241 DI": {"hp": 50, "category": "MF"},
    "MF-1035 DI - J- MAHASHAKTI": {"hp": 35, "category": "MF"},
    "MF-245 DI": {"hp": 55, "category": "MF"},
    "MF-7250 DI": {"hp": 75, "category": "MF"},
    "MF-1030 DI": {"hp": 30, "category": "MF"},
    "MF-9000 DI": {"hp": 90, "category": "MF"},
    "MAHINDRA YUVO TECH + 405 DI": {"hp": 39, "category": "Mahindra"},
    "Mahindra 575 DI": {"hp": 50, "category": "Mahindra"},
    "Mahindra 475 DI": {"hp": 42, "category": "Mahindra"},
    "Mahindra 265 DI": {"hp": 35, "category": "Mahindra"},
}

# Known Dealer Patterns (fuzzy matching base)
# ADD YOUR DEALERS HERE
KNOWN_DEALERS = [
    "National Tractor Sales",
    "K D Tractors",
    "Shyam Tractors",
    "ABC Tractors Pvt Ltd",
    "Local Tractor Dealer",
    "IDFC First Bank",
    "Mahindra Tractor Sales",
    "Massey Ferguson Dealer",
]

# ENHANCED: Regex patterns for field extraction
PATTERNS = {
    "hp": r"(\d+)\s*(?:HP|hp|Hp|H\.P\.)",
    "cost": r"(?:Rs\.?\s*|₹\s*)?(\d{1,3}(?:,\d{3})*|\d+)",
    "model": r"(?:Model|MODEL|model)\s*[:\-]?\s*([^\n]+?)(?:\n|$|,)",
    "dealer": r"(?:Dealer|DEALER|dealer)\s*[:\-]?\s*([^\n]+?)(?:\n|$)",
    "amount": r"(?:Total|TOTAL|Amount|AMOUNT|Cost|COST)\s*[:\-]?\s*(?:Rs\.?|₹)?\s*([0-9,]+)",
}

# ENHANCED: Confidence thresholds (lowered for better recall)
CONFIDENCE_THRESHOLDS = {
    "dealer_name": 0.70,    # Lowered from 0.75
    "model_name": 0.75,     # Lowered from 0.85
    "horse_power": 0.85,    # Lowered from 0.90
    "asset_cost": 0.85,     # Lowered from 0.90
    "signature": 0.65,      # Lowered from 0.70
    "stamp": 0.65,          # Lowered from 0.70
}

# ENHANCED: OCR Settings for PaddleOCR
OCR_CONFIG = {
    "use_angle_cls": True,      # Auto-rotate text
    "use_gpu": False,           # Set to True if GPU available
    "lang": ["en", "hi"],       # English & Hindi (add 'gu' for Gujarati)
    "det_db_thresh": 0.3,       # Detection threshold (lower = more sensitive)
    "det_db_box_thresh": 0.5,   # Box threshold
    "rec_batch_num": 6,         # Batch size for recognition
}

# NEW: Preprocessing configurations
PREPROCESSING_CONFIGS = {
    "fast": {
        "resize": True,
        "denoise": True,
        "clahe": False,
        "sharpen": False,
        "deskew": False,
        "remove_shadow": False
    },
    "standard": {
        "resize": True,
        "denoise": True,
        "clahe": True,
        "sharpen": False,
        "deskew": True,
        "remove_shadow": False
    },
    "aggressive": {
        "resize": True,
        "denoise": True,
        "clahe": True,
        "sharpen": True,
        "deskew": True,
        "remove_shadow": True
    }
}

# NEW: Field extraction strategies priority
EXTRACTION_STRATEGIES = {
    "dealer_name": ["keyword_search", "fuzzy_match", "company_pattern"],
    "model_name": ["pattern_match", "fuzzy_match", "keyword_search"],
    "horse_power": ["explicit_pattern", "model_lookup", "numeric_guess"],
    "asset_cost": ["keyword_number", "largest_number", "currency_pattern"]
}

# Performance settings
PERFORMANCE_CONFIG = {
    "max_image_dimension": 3000,  # Max width/height in pixels
    "target_dpi": 300,             # Target DPI for PDF conversion
    "ocr_timeout": 30,             # Timeout for OCR in seconds
    "enable_caching": False,       # Cache OCR results (not implemented yet)
}

# Validation ranges
VALIDATION_RANGES = {
    "horse_power": (10, 200),       # Min/max reasonable HP
    "asset_cost": (100000, 10000000),  # Min/max reasonable cost (₹)
}
