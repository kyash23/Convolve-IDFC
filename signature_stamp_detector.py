# signature_stamp_detector.py
"""
Detect signature and stamp presence & locations
"""
import cv2
import numpy as np
from typing import Dict, Tuple, List
from utils import find_large_contours

class SignatureStampDetector:
    """Detect signatures and stamps in documents"""
    
    def __init__(self, min_signature_area: int = 500, min_stamp_area: int = 1000):
        """
        Initialize detector
        
        Args:
            min_signature_area: Minimum contour area for signature
            min_stamp_area: Minimum contour area for stamp
        """
        self.min_signature_area = min_signature_area
        self.min_stamp_area = min_stamp_area
    
    def detect_signature(self, image: np.ndarray) -> Tuple[bool, List[int], float]:
        """
        Detect signature in image
        Signatures typically have specific visual characteristics
        
        Returns:
            (is_present, bbox, confidence)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find contours with inverted threshold (signatures are darker lines)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        contours = find_large_contours(image, min_area=self.min_signature_area)
        
        if not contours:
            return False, [0, 0, 0, 0], 0.0
        
        # Get largest contour (most likely signature)
        largest = contours[0]
        bbox = largest['bbox']
        
        # Calculate confidence based on contour properties
        # Signatures typically have specific aspect ratios (wider than tall)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        aspect_ratio = width / height if height > 0 else 0
        
        # Good signature aspect ratio is typically 2:1 to 5:1
        confidence = min(1.0, 1.0 - abs(aspect_ratio - 3.0) / 5.0)
        
        return True, bbox, confidence
    
    def detect_stamp(self, image: np.ndarray) -> Tuple[bool, List[int], float]:
        """
        Detect stamp (usually circular or polygonal with text)
        
        Returns:
            (is_present, bbox, confidence)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stamp_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Stamps typically have reasonable size
            if area < self.min_stamp_area:
                continue
            
            # Fit circle to contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Stamps are often circular (circularity ~0.7-1.0)
            if circularity > 0.6:
                x_min = int(max(0, x - radius))
                y_min = int(max(0, y - radius))
                x_max = int(x + radius)
                y_max = int(y + radius)
                
                stamp_candidates.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'circularity': circularity,
                    'area': area
                })
        
        if not stamp_candidates:
            return False, [0, 0, 0, 0], 0.0
        
        # Get best candidate (highest circularity)
        best = max(stamp_candidates, key=lambda x: x['circularity'])
        
        return True, best['bbox'], min(1.0, best['circularity'])
    
    def detect_both(self, image: np.ndarray) -> Dict:
        """
        Detect both signature and stamp
        
        Returns:
            Dictionary with both detections
        """
        sig_present, sig_bbox, sig_conf = self.detect_signature(image)
        stamp_present, stamp_bbox, stamp_conf = self.detect_stamp(image)
        
        return {
            "signature": {
                "present": sig_present,
                "bbox": sig_bbox,
                "confidence": sig_conf
            },
            "stamp": {
                "present": stamp_present,
                "bbox": stamp_bbox,
                "confidence": stamp_conf
            }
        }

class VisualElementValidator:
    """Validate detected visual elements"""
    
    @staticmethod
    def validate_bbox(bbox: List[int], image_shape: Tuple) -> bool:
        """Check if bounding box is valid"""
        if len(bbox) != 4:
            return False
        
        x_min, y_min, x_max, y_max = bbox
        height, width = image_shape[:2]
        
        # Check bounds
        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
            return False
        
        # Check dimensions
        if x_max <= x_min or y_max <= y_min:
            return False
        
        return True
    
    @staticmethod
    def calculate_region_density(image: np.ndarray, bbox: List[int]) -> float:
        """
        Calculate ink density in a region (0-1)
        Used to validate if region actually contains signature/stamp
        """
        x_min, y_min, x_max, y_max = bbox
        region = image[y_min:y_max, x_min:x_max]
        
        if region.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Count dark pixels (below 150 intensity)
        dark_pixels = np.sum(gray < 150)
        total_pixels = gray.size
        
        density = dark_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return float(density)
