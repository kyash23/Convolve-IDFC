# pdf_handler.py - UPDATED WITH ENHANCED PREPROCESSING
"""
PDF Processing & Advanced Image Preprocessing
Major improvements for better OCR accuracy
"""
import os
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2

class PDFHandler:
    """Handle PDF to image conversion"""
    
    def __init__(self, dpi: int = 300):
        """Initialize PDF handler with DPI setting"""
        self.dpi = dpi
    
    def pdf_to_images(self, pdf_path: str) -> list:
        """
        Convert PDF to list of PIL Images
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of PIL Image objects
        """
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=self.dpi)
            
            print(f"✅ Converted PDF to {len(images)} page(s)")
            return images
        
        except Exception as e:
            print(f"❌ Error converting PDF: {str(e)}")
            return []
    
    def image_to_array(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array"""
        return np.array(image)
    
    def save_image(self, image: Image.Image, output_path: str):
        """Save PIL Image to disk"""
        image.save(output_path)
        print(f"✅ Saved image to {output_path}")


class ImagePreprocessor:
    """ENHANCED - Advanced preprocessing for better OCR"""
    
    @staticmethod
    def enhance_for_ocr(image_array: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        IMPROVED: Multi-stage preprocessing pipeline for better OCR
        
        Args:
            image_array: Input image
            aggressive: Use aggressive preprocessing for poor quality images
        
        Returns:
            Enhanced image optimized for OCR
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()
        
        # CRITICAL: Resize if image is too small
        # OCR works MUCH better on larger images
        height, width = gray.shape
        if height < 1200:
            scale = 1500 / height  # Target height of 1500px
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"📐 Resized image: {width}x{height} → {new_width}x{new_height}")
        
        # Step 1: Remove noise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Step 2: Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        if aggressive:
            print("⚡ Using aggressive preprocessing...")
            
            # Step 3: Sharpen image (for blurry scans)
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Step 4: Morphological operations to clean text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Step 5: Adaptive thresholding for very poor quality
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        return enhanced
    
    @staticmethod
    def auto_deskew(image_array: np.ndarray) -> np.ndarray:
        """
        NEW: Automatically detect and correct image skew/rotation
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 0:
            # Calculate median angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            median_angle = np.median(angles)
            
            # Rotate if significant skew detected
            if abs(median_angle) > 0.5:
                print(f"🔄 Deskewing image: {median_angle:.2f}°")
                (h, w) = image_array.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(
                    image_array, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
        
        return image_array
    
    @staticmethod
    def remove_shadows(image_array: np.ndarray) -> np.ndarray:
        """
        NEW: Remove shadows from scanned documents
        Helps with photos of documents taken in poor lighting
        """
        if len(image_array.shape) == 3:
            rgb_planes = cv2.split(image_array)
            result_planes = []
            
            for plane in rgb_planes:
                dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg = cv2.medianBlur(dilated, 21)
                diff = 255 - cv2.absdiff(plane, bg)
                result_planes.append(diff)
            
            return cv2.merge(result_planes)
        else:
            dilated = cv2.dilate(image_array, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            return 255 - cv2.absdiff(image_array, bg)
    
    @staticmethod
    def preprocess_pipeline(image_array: np.ndarray, quality: str = 'standard') -> np.ndarray:
        """
        NEW: Complete preprocessing pipeline with quality levels
        
        Args:
            image_array: Input image
            quality: 'fast', 'standard', or 'aggressive'
        
        Returns:
            Preprocessed image
        """
        preprocessor = ImagePreprocessor()
        
        if quality == 'fast':
            # Quick preprocessing - just resize and denoise
            return preprocessor.enhance_for_ocr(image_array, aggressive=False)
        
        elif quality == 'standard':
            # Standard pipeline: deskew + enhance
            deskewed = preprocessor.auto_deskew(image_array)
            enhanced = preprocessor.enhance_for_ocr(deskewed, aggressive=False)
            return enhanced
        
        elif quality == 'aggressive':
            # Full pipeline: shadow removal + deskew + aggressive enhance
            no_shadow = preprocessor.remove_shadows(image_array)
            deskewed = preprocessor.auto_deskew(no_shadow)
            enhanced = preprocessor.enhance_for_ocr(deskewed, aggressive=True)
            return enhanced
        
        else:
            return preprocessor.enhance_for_ocr(image_array, aggressive=False)
    
    @staticmethod
    def binarize(image_array: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Binarize image for better text detection
        
        Args:
            image_array: Input image
            method: 'otsu' or 'adaptive'
        
        Returns:
            Binary image
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # adaptive
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        return binary
