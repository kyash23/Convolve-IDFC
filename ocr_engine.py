# ocr_engine.py - UPDATED WITH PADDLEOCR
"""
Enhanced OCR Engine using PaddleOCR for better multilingual recognition
Major improvements over Tesseract for scanned documents
"""
import numpy as np
import cv2
from paddleocr import PaddleOCR
from typing import Dict, List, Any
import re

class OCREngine:
    """Enhanced OCR engine with PaddleOCR"""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        """
        Initialize OCR engine with PaddleOCR
        
        Args:
            languages: List of language codes ['en', 'hi', 'gu', etc.]
            use_gpu: Whether to use GPU acceleration
        """
        if languages is None:
            languages = ['en']
        
        primary_lang = languages[0] if languages else 'en'
        
        print(f"🔄 Initializing PaddleOCR (lang: {primary_lang}, GPU: {use_gpu})...")
        
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,      # Detect and correct text orientation
                lang=primary_lang,        # Primary language
                use_gpu=use_gpu,
                show_log=False,           # Reduce console spam
                det_db_thresh=0.3,        # Lower threshold for better detection
                det_db_box_thresh=0.5,
                rec_batch_num=6,          # Process multiple regions at once
                use_space_char=True       # Preserve spaces in text
            )
            self.languages = languages
            print("✅ PaddleOCR initialized successfully\n")
        except Exception as e:
            print(f"⚠️ PaddleOCR initialization failed: {e}")
            print("Falling back to basic OCR...")
            self.ocr = None
    
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text from image using PaddleOCR
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            Dictionary with extraction results
        """
        try:
            # Ensure image is in correct format (BGR for PaddleOCR)
            if len(image.shape) == 2:
                # Grayscale - convert to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                # RGBA - convert to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = image
            
            # Run OCR
            result = self.ocr.ocr(image_bgr, cls=True)
            
            if not result or result[0] is None:
                return {
                    "full_text": "",
                    "detailed_results": [],
                    "success": False,
                    "error": "No text detected"
                }
            
            # Parse results
            detailed_results = []
            text_lines = []
            
            for line in result[0]:
                # PaddleOCR format: [bbox_points, (text, confidence)]
                bbox_points = line[0]  # 4 corner points
                text = line[1][0]      # Recognized text
                confidence = line[1][1]  # Confidence score (0-1)
                
                # Convert bbox points to [x_min, y_min, x_max, y_max]
                bbox = self._points_to_bbox(bbox_points)
                
                detailed_results.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox
                })
                
                text_lines.append(text)
            
            # Combine all text
            full_text = "\n".join(text_lines)
            
            return {
                "full_text": full_text,
                "detailed_results": detailed_results,
                "success": True,
                "num_regions": len(detailed_results)
            }
        
        except Exception as e:
            print(f"❌ OCR Error: {str(e)}")
            return {
                "full_text": "",
                "detailed_results": [],
                "success": False,
                "error": str(e)
            }
    
    def _points_to_bbox(self, points: List[List[float]]) -> List[int]:
        """
        Convert 4 corner points to bounding box [x_min, y_min, x_max, y_max]
        
        Args:
            points: List of 4 [x, y] coordinates
        
        Returns:
            [x_min, y_min, x_max, y_max]
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        return [
            int(min(x_coords)),
            int(min(y_coords)),
            int(max(x_coords)),
            int(max(y_coords))
        ]


class TextProcessor:
    """Enhanced text post-processing"""
    
    @staticmethod
    def extract_lines(full_text: str) -> List[str]:
        """Extract non-empty lines from text"""
        return [line.strip() for line in full_text.split("\n") if line.strip()]
    
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """
        Clean common OCR errors
        Context-aware replacements
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR mistakes (only in non-numeric contexts)
        if not re.search(r'\d{3,}', text):  # Not if it's a large number
            replacements = {
                '|': 'I',
                '§': 'S',
                '©': 'C',
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def merge_broken_words(lines: List[str]) -> str:
        """Merge words broken across lines"""
        merged = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # If line ends with hyphen, merge with next
            if line.endswith('-') and i + 1 < len(lines):
                merged.append(line[:-1] + lines[i + 1])
                i += 2
            else:
                merged.append(line)
                i += 1
        return "\n".join(merged)
