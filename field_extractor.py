# field_extractor.py - UPDATED WITH ENHANCED EXTRACTION
"""
ENHANCED Field extraction logic for invoice documents
Improvements:
- More robust pattern matching
- Better fuzzy matching
- Multi-strategy extraction with fallbacks
- Improved numeric extraction
"""
import re
from typing import Dict, Tuple, Any, List
import numpy as np
from config import TRACTOR_MODELS, KNOWN_DEALERS, PATTERNS
from utils import (clean_text, extract_cost_value, fuzzy_match_dealer, 
                   fuzzy_match_model, extract_numbers)

class FieldExtractor:
    """ENHANCED - Extract specific fields from OCR text"""
    
    def __init__(self):
        self.tractor_models = TRACTOR_MODELS
        self.known_dealers = KNOWN_DEALERS
        self.patterns = PATTERNS
    
    def extract_dealer_name(self, full_text: str, detailed_results: List[Dict] = None) -> Tuple[str | None, float]:
        """
        ENHANCED: Extract dealer name with multiple strategies
        
        Returns:
            (dealer_name, confidence)
        """
        # Strategy 1: Search for "Dealer" keyword and surrounding text
        dealer_patterns = [
            r'(?:Dealer|DEALER|dealer)[\s:]*([^\n]+)',
            r'(?:By|BY|by)[\s:]*([^\n]+)',
            r'(?:Authorized|AUTHORIZED|authorized)[\s:]*([^\n]+)',
        ]
        
        candidates = []
        
        for pattern in dealer_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                text = match.group(1).strip()
                # Clean up common OCR artifacts
                text = re.sub(r'\s+', ' ', text)
                text = text.split('\n')[0]  # Take only first line
                
                if len(text) > 3:  # Minimum length
                    # Fuzzy match against known dealers
                    matched_name, confidence = fuzzy_match_dealer(text, self.known_dealers, threshold=70)
                    if matched_name:
                        candidates.append((matched_name, confidence))
                    else:
                        # Use extracted text with lower confidence if no match
                        candidates.append((clean_text(text), 0.65))
        
        # Strategy 2: Look in detailed OCR results for dealer-like entities
        if detailed_results and not candidates:
            for item in detailed_results[:10]:  # Check first 10 text regions
                text = item.get('text', '')
                # Look for company-like names (capitalized, contains Ltd/Pvt/Inc)
                if re.search(r'(Ltd|Pvt|Inc|LLC|Corp)', text, re.IGNORECASE):
                    matched_name, confidence = fuzzy_match_dealer(text, self.known_dealers, threshold=70)
                    if matched_name:
                        candidates.append((matched_name, confidence))
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[1]
        
        return None, 0.0
    
    def extract_model_name(self, full_text: str, detailed_results: List[Dict] = None) -> Tuple[str | None, float]:
        """
        ENHANCED: Extract tractor model name with better pattern matching
        
        Returns:
            (model_name, confidence)
        """
        # Strategy 1: Model keyword patterns
        model_patterns = [
            r'(?:Model|MODEL|model)[\s:]*([^\n,]+)',
            r'(?:Tractor|TRACTOR|tractor)[\s:]*([^\n,]+)',
            r'([A-Z]{2,}[-\s]?\d+[\s]+DI)',  # Pattern like "MF-1035 DI" or "MF 1035 DI"
            r'(Mahindra\s+\w+\s+\d+)',  # "Mahindra YUVO 405"
        ]
        
        candidates = []
        
        for pattern in model_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                candidate_text = match.group(1).strip()
                candidate_text = re.sub(r'\s+', ' ', candidate_text)
                
                if len(candidate_text) > 3:
                    candidates.append(candidate_text)
        
        # Strategy 2: Try fuzzy matching against master models
        best_match = None
        best_confidence = 0.0
        
        for candidate in candidates:
            matched_model, confidence = fuzzy_match_model(candidate, self.tractor_models, threshold=70)
            if matched_model and confidence > best_confidence:
                best_match = matched_model
                best_confidence = confidence
        
        if best_match:
            return best_match, best_confidence
        
        # Strategy 3: Look for model-like patterns in detailed results
        if detailed_results:
            for item in detailed_results:
                text = item.get('text', '')
                # Check if matches any known model pattern
                for model_key in self.tractor_models.keys():
                    if re.search(re.escape(model_key.split()[0]), text, re.IGNORECASE):
                        matched_model, confidence = fuzzy_match_model(text, self.tractor_models, threshold=65)
                        if matched_model and confidence > best_confidence:
                            best_match = matched_model
                            best_confidence = confidence
        
        # Fallback: Return best candidate even without exact match
        if not best_match and candidates:
            best_candidate = max(candidates, key=len)
            return clean_text(best_candidate), 0.70
        
        return best_match, best_confidence
    
    def extract_horse_power(self, full_text: str, model_name: str = None) -> Tuple[float | None, float]:
        """
        ENHANCED: Extract horse power with multiple strategies
        
        Returns:
            (hp_value, confidence)
        """
        # Strategy 1: Explicit HP pattern
        hp_patterns = [
            r'(\d+)\s*(?:HP|hp|Hp|H\.P\.|horse\s*power)',
            r'(?:HP|hp|Hp)[\s:]*(\d+)',
            r'(\d+)\s*(?:Horse\s*Power|HORSE\s*POWER)',
        ]
        
        for pattern in hp_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                try:
                    hp_value = float(match.group(1))
                    # Sanity check: reasonable HP range for tractors
                    if 10 <= hp_value <= 200:
                        return hp_value, 0.95
                except (ValueError, IndexError):
                    continue
        
        # Strategy 2: Get HP from model master data
        if model_name and model_name in self.tractor_models:
            hp_value = float(self.tractor_models[model_name].get('hp', 0))
            if hp_value > 0:
                return hp_value, 0.90
        
        # Strategy 3: Look for standalone numbers that might be HP
        numbers = re.findall(r'\b(\d{2})\b', full_text)
        for num in numbers:
            try:
                hp_candidate = float(num)
                # Check if in reasonable HP range
                if 20 <= hp_candidate <= 150:
                    # Low confidence - just a guess
                    return hp_candidate, 0.60
            except ValueError:
                continue
        
        return None, 0.0
    
    def extract_asset_cost(self, full_text: str, detailed_results: List[Dict] = None) -> Tuple[float | None, float]:
        """
        ENHANCED: Extract asset cost with better pattern matching
        
        Returns:
            (cost_value, confidence)
        """
        # Strategy 1: Look for cost-related keywords with numbers
        cost_patterns = [
            r'(?:Total|TOTAL|total|Cost|COST|cost|Amount|AMOUNT|amount|Price|PRICE|price)[\s:]*(?:Rs\.?|₹)?[\s]*([0-9,]+)',
            r'(?:Rs\.?|₹)\s*([0-9,]+)',
            r'(?:INR|Rupees)[\s:]*([0-9,]+)',
        ]
        
        candidates = []
        
        for pattern in cost_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                # Extract number (remove commas)
                value_str = match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    # Reasonable tractor cost: 100,000 to 10,000,000
                    if 100000 <= value <= 10000000:
                        # Higher confidence if keyword present
                        keyword = match.group(0).lower()
                        if 'total' in keyword or 'amount' in keyword:
                            confidence = 0.95
                        elif 'cost' in keyword or 'price' in keyword:
                            confidence = 0.90
                        else:
                            confidence = 0.80
                        
                        candidates.append((value, confidence))
                except (ValueError, IndexError):
                    continue
        
        # Strategy 2: Find largest number in reasonable range
        if not candidates:
            all_numbers = re.findall(r'([0-9,]{5,})', full_text)
            for num_str in all_numbers:
                try:
                    value = float(num_str.replace(',', ''))
                    if 100000 <= value <= 10000000:
                        candidates.append((value, 0.70))
                except ValueError:
                    continue
        
        # Strategy 3: Check detailed results for large numbers
        if not candidates and detailed_results:
            for item in detailed_results:
                text = item.get('text', '')
                numbers = re.findall(r'([0-9,]{5,})', text)
                for num_str in numbers:
                    try:
                        value = float(num_str.replace(',', ''))
                        if 100000 <= value <= 10000000:
                            candidates.append((value, 0.65))
                    except ValueError:
                        continue
        
        # Return highest confidence or largest value
        if candidates:
            # Sort by confidence, then by value
            candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
            return candidates[0]
        
        return None, 0.0
    
    def extract_all_fields(self, full_text: str, detailed_ocr_results: list = None) -> Dict[str, Any]:
        """
        ENHANCED: Extract all fields with detailed_results support
        
        Args:
            full_text: Full OCR text
            detailed_ocr_results: Detailed OCR results with bboxes and confidence
        
        Returns:
            Dictionary with all extracted fields
        """
        # Clean the full text first
        full_text = clean_text(full_text)
        
        # Extract fields sequentially
        dealer_name, dealer_conf = self.extract_dealer_name(full_text, detailed_ocr_results)
        model_name, model_conf = self.extract_model_name(full_text, detailed_ocr_results)
        horse_power, hp_conf = self.extract_horse_power(full_text, model_name)
        asset_cost, cost_conf = self.extract_asset_cost(full_text, detailed_ocr_results)
        
        # Boost confidence if model and HP match from master data
        if model_name and model_name in self.tractor_models:
            expected_hp = self.tractor_models[model_name].get('hp')
            if horse_power and expected_hp and abs(horse_power - expected_hp) <= 5:
                hp_conf = min(1.0, hp_conf + 0.05)  # Small confidence boost
                model_conf = min(1.0, model_conf + 0.05)
        
        # Return structure
        return {
            "dealer_name": {
                "value": dealer_name,
                "confidence": round(dealer_conf, 3)
            },
            "model_name": {
                "value": model_name,
                "confidence": round(model_conf, 3)
            },
            "horse_power": {
                "value": horse_power,
                "confidence": round(hp_conf, 3)
            },
            "asset_cost": {
                "value": asset_cost,
                "confidence": round(cost_conf, 3)
            }
        }
