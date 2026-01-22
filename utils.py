# utils.py
"""
Utility functions for invoice extraction
"""
import json
import re
from typing import Tuple, List, Dict, Any
from fuzzywuzzy import fuzz
import cv2
import numpy as np

def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text"""
    return re.findall(r'\d+', text)

def extract_cost_value(text: str) -> Tuple[float | None, float]:
    """
    Extract numeric cost from text
    Returns: (value, confidence)
    """
    # Remove currency symbols
    cleaned = re.sub(r'[Rs₹£$€]', '', text).strip()
    
    # Extract numbers
    numbers = re.findall(r'\d+', cleaned)
    
    if numbers:
        # Assume last number is the cost
        cost = float(numbers[-1])
        confidence = 0.95 if len(numbers) <= 2 else 0.80
        return cost, confidence
    
    return None, 0.0

def fuzzy_match_dealer(text: str, master_list: List[str], threshold: int = 85) -> Tuple[str | None, float]:
    """
    Fuzzy match dealer name against master list
    Returns: (matched_name, confidence_score)
    """
    text = clean_text(text).upper()
    
    best_match = None
    best_score = 0
    
    for dealer in master_list:
        score = fuzz.token_set_ratio(text, dealer.upper())
        if score > best_score:
            best_score = score
            best_match = dealer
    
    # Normalize to 0-1 range
    confidence = best_score / 100.0 if best_match else 0.0
    
    if confidence >= threshold / 100:
        return best_match, confidence
    return None, confidence

def fuzzy_match_model(text: str, master_dict: Dict[str, Any], threshold: int = 80) -> Tuple[str | None, float]:
    """
    Fuzzy match model name against master dictionary
    Returns: (matched_model_name, confidence_score)
    """
    text = clean_text(text).upper()
    
    best_match = None
    best_score = 0
    
    for model in master_dict.keys():
        score = fuzz.partial_token_set_ratio(text, model.upper())
        if score > best_score:
            best_score = score
            best_match = model
    
    confidence = best_score / 100.0 if best_match else 0.0
    
    if confidence >= threshold / 100:
        return best_match, confidence
    return None, confidence

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    box format: [x_min, y_min, x_max, y_max]
    """
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    
    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        return 0.0
    
    inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def find_large_contours(image: np.ndarray, min_area: int = 500) -> List[Dict[str, Any]]:
    """
    Find large contours in image (for signature/stamp detection)
    Returns: List of contours with bounding boxes and areas
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            results.append({
                'bbox': [x, y, x+w, y+h],
                'area': area,
                'contour': contour
            })
    
    return sorted(results, key=lambda x: x['area'], reverse=True)

def json_serialize(obj):
    """Custom JSON serializer for numpy types"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def save_json_output(data: Dict[str, Any], filepath: str):
    """Save extraction results as JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=json_serialize, ensure_ascii=False)

def load_json_output(filepath: str) -> Dict[str, Any]:
    """Load JSON output"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
