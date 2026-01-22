# executable.py - UPDATED WITH ENHANCED PIPELINE
"""
Main Invoice Field Extraction Pipeline - IMPROVED VERSION
Major changes:
- Better preprocessing pipeline
- Enhanced error handling
- Quality-based preprocessing options
- Better logging and diagnostics
"""
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2

# Import custom modules
from pdf_handler import PDFHandler, ImagePreprocessor
from ocr_engine import OCREngine, TextProcessor
from field_extractor import FieldExtractor
from signature_stamp_detector import SignatureStampDetector, VisualElementValidator
from config import CONFIDENCE_THRESHOLDS, OCR_CONFIG
from utils import save_json_output, load_json_output

class InvoiceExtractionPipeline:
    """ENHANCED - Complete invoice extraction pipeline"""
    
    def __init__(self, output_dir: str = "output", preprocessing_quality: str = "standard"):
        """
        Initialize pipeline
        
        Args:
            output_dir: Directory for output files
            preprocessing_quality: 'fast', 'standard', or 'aggressive'
        """
        self.output_dir = output_dir
        self.preprocessing_quality = preprocessing_quality
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        print("="*60)
        print("INITIALIZING ENHANCED INVOICE EXTRACTION PIPELINE")
        print("="*60)
        
        print("\n[1/5] Initializing PDF handler...")
        self.pdf_handler = PDFHandler(dpi=300)
        
        print("[2/5] Initializing OCR engine (PaddleOCR)...")
        self.ocr_engine = OCREngine(
            languages=OCR_CONFIG.get('lang', ['en']),
            use_gpu=OCR_CONFIG.get('use_gpu', False)
        )
        
        print("[3/5] Initializing field extractor...")
        self.field_extractor = FieldExtractor()
        
        print("[4/5] Initializing signature/stamp detector...")
        self.signature_detector = SignatureStampDetector()
        
        print("[5/5] Initializing image preprocessor...")
        self.image_preprocessor = ImagePreprocessor()
        
        print("\n" + "="*60)
        print("✅ PIPELINE READY")
        print(f"📊 Preprocessing quality: {preprocessing_quality}")
        print("="*60 + "\n")
    
    def process_pdf(self, pdf_path: str, doc_id: str = None, 
                    retry_on_failure: bool = True) -> Dict[str, Any]:
        """
        ENHANCED: Process single PDF with retry logic
        
        Args:
            pdf_path: Path to PDF file
            doc_id: Optional document identifier
            retry_on_failure: Retry with aggressive preprocessing if OCR fails
        
        Returns:
            Extraction results as dictionary
        """
        start_time = time.time()
        
        if doc_id is None:
            doc_id = Path(pdf_path).stem
        
        print(f"\n{'='*60}")
        print(f"📄 Processing: {doc_id}")
        print(f"📂 File: {pdf_path}")
        print(f"{'='*60}")
        
        # Step 1: Convert PDF to images
        print("\n[1/6] Converting PDF to images...")
        images = self.pdf_handler.pdf_to_images(pdf_path)
        
        if not images:
            return self._error_result(doc_id, "PDF conversion failed", start_time)
        
        # Use first page for extraction
        image_pil = images[0]
        image_array = np.array(image_pil)
        print(f"   Image size: {image_array.shape[1]}x{image_array.shape[0]}")
        
        # Step 2: ENHANCED PREPROCESSING
        print(f"\n[2/6] Preprocessing image (quality: {self.preprocessing_quality})...")
        try:
            processed_image = self.image_preprocessor.preprocess_pipeline(
                image_array, 
                quality=self.preprocessing_quality
            )
            print(f"   ✅ Preprocessing complete")
        except Exception as e:
            print(f"   ⚠️ Preprocessing error: {e}")
            processed_image = image_array
        
        # Step 3: OCR extraction
        print("\n[3/6] Extracting text with OCR...")
        ocr_results = self.ocr_engine.extract_text(processed_image)
        
        # RETRY LOGIC: If OCR fails and retry enabled, try aggressive preprocessing
        if not ocr_results['success'] and retry_on_failure and self.preprocessing_quality != 'aggressive':
            print("   ⚠️ Initial OCR failed, retrying with aggressive preprocessing...")
            processed_image = self.image_preprocessor.preprocess_pipeline(
                image_array, 
                quality='aggressive'
            )
            ocr_results = self.ocr_engine.extract_text(processed_image)
        
        if not ocr_results['success']:
            return self._error_result(doc_id, f"OCR extraction failed: {ocr_results.get('error', 'Unknown')}", start_time)
        
        full_text = ocr_results['full_text']
        detailed_results = ocr_results['detailed_results']
        
        print(f"   ✅ Extracted {len(detailed_results)} text regions")
        print(f"   📝 Total characters: {len(full_text)}")
        
        # Debug: Print first 200 chars of extracted text
        preview = full_text[:200].replace('\n', ' ')
        print(f"   🔍 Preview: {preview}...")
        
        # Step 4: Field extraction
        print("\n[4/6] Extracting fields...")
        field_results = self.field_extractor.extract_all_fields(
            full_text, detailed_results
        )
        
        # Print field extraction results
        for field_name, field_data in field_results.items():
            value = field_data['value']
            conf = field_data['confidence']
            status = "✅" if conf > 0.75 else "⚠️"
            print(f"   {status} {field_name}: {value} (conf: {conf:.2f})")
        
        # Step 5: Signature & Stamp detection
        print("\n[5/6] Detecting signature and stamp...")
        visual_results = self.signature_detector.detect_both(image_array)
        
        # Validate visual detections
        validator = VisualElementValidator()
        for key in ['signature', 'stamp']:
            if visual_results[key]['present']:
                bbox = visual_results[key]['bbox']
                if not validator.validate_bbox(bbox, image_array.shape):
                    print(f"   ⚠️ {key.capitalize()} bbox validation failed")
                    visual_results[key]['present'] = False
                    visual_results[key]['confidence'] = 0.0
                else:
                    print(f"   ✅ {key.capitalize()} detected (conf: {visual_results[key]['confidence']:.2f})")
            else:
                print(f"   ❌ {key.capitalize()} not found")
        
        # Step 6: Compile results
        print("\n[6/6] Compiling results...")
        
        processing_time = time.time() - start_time
        
        results = {
            "doc_id": doc_id,
            "fields": {
                "dealer_name": field_results['dealer_name']['value'],
                "model_name": field_results['model_name']['value'],
                "horse_power": field_results['horse_power']['value'],
                "asset_cost": field_results['asset_cost']['value'],
                "signature": {
                    "present": visual_results['signature']['present'],
                    "bbox": visual_results['signature']['bbox']
                },
                "stamp": {
                    "present": visual_results['stamp']['present'],
                    "bbox": visual_results['stamp']['bbox']
                }
            },
            "confidence": {
                "dealer_name": field_results['dealer_name']['confidence'],
                "model_name": field_results['model_name']['confidence'],
                "horse_power": field_results['horse_power']['confidence'],
                "asset_cost": field_results['asset_cost']['confidence'],
                "signature": visual_results['signature']['confidence'],
                "stamp": visual_results['stamp']['confidence']
            },
            "processing_time_sec": round(processing_time, 2),
            "cost_estimate_usd": self._estimate_cost(processing_time),
            "overall_confidence": self._calculate_overall_confidence(
                field_results, visual_results
            ),
            "preprocessing_quality": self.preprocessing_quality,
            "success": True
        }
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def process_batch(self, pdf_directory: str, output_file: str = "batch_results.json"):
        """
        Process multiple PDFs in a directory
        
        Args:
            pdf_directory: Directory containing PDF files
            output_file: Output JSON file for results
        """
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING: {len(pdf_files)} documents")
        print(f"{'='*60}\n")
        
        results = []
        successful = 0
        failed = 0
        total_time = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}...")
            
            try:
                result = self.process_pdf(str(pdf_file))
                results.append(result)
                
                if result.get('success', True):
                    successful += 1
                    total_time += result.get('processing_time_sec', 0)
                else:
                    failed += 1
            
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")
                failed += 1
                results.append({
                    "doc_id": pdf_file.stem,
                    "error": str(e),
                    "success": False
                })
        
        # Save batch results
        output_path = os.path.join(self.output_dir, output_file)
        save_json_output(results, output_path)
        
        # Print statistics
        avg_time = total_time / successful if successful > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}/{len(pdf_files)} ({successful/len(pdf_files)*100:.1f}%)")
        print(f"❌ Failed: {failed}/{len(pdf_files)}")
        print(f"⏱️  Average time: {avg_time:.2f}s per document")
        print(f"📁 Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _error_result(self, doc_id: str, error_msg: str, start_time: float) -> Dict:
        """Create error result structure"""
        return {
            "doc_id": doc_id,
            "error": error_msg,
            "success": False,
            "processing_time_sec": round(time.time() - start_time, 2),
            "fields": {
                "dealer_name": None,
                "model_name": None,
                "horse_power": None,
                "asset_cost": None,
                "signature": {"present": False, "bbox": [0, 0, 0, 0]},
                "stamp": {"present": False, "bbox": [0, 0, 0, 0]}
            },
            "confidence": {
                "dealer_name": 0.0,
                "model_name": 0.0,
                "horse_power": 0.0,
                "asset_cost": 0.0,
                "signature": 0.0,
                "stamp": 0.0
            },
            "overall_confidence": 0.0
        }
    
    def _calculate_overall_confidence(self, field_results: Dict, visual_results: Dict) -> float:
        """Calculate overall extraction confidence"""
        field_confs = [
            field_results['dealer_name']['confidence'],
            field_results['model_name']['confidence'],
            field_results['horse_power']['confidence'],
            field_results['asset_cost']['confidence']
        ]
        
        visual_confs = [
            visual_results['signature']['confidence'],
            visual_results['stamp']['confidence']
        ]
        
        # Weight: 70% field extraction, 30% visual elements
        field_avg = sum(field_confs) / len(field_confs) if field_confs else 0.0
        visual_avg = sum(visual_confs) / len(visual_confs) if visual_confs else 0.0
        
        overall = (0.7 * field_avg) + (0.3 * visual_avg)
        
        return round(overall, 3)
    
    def _estimate_cost(self, processing_time: float) -> float:
        """Estimate inference cost (CPU-based model)"""
        # Rough estimate: $0.001 per second on shared CPU tier
        cost = processing_time * 0.001
        return round(cost, 5)
    
    def _print_summary(self, results: Dict):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("📊 EXTRACTION RESULTS")
        print("="*60)
        
        fields = results['fields']
        confidence = results['confidence']
        
        print(f"\n📋 TEXT FIELDS:")
        print(f"  Dealer Name:    {fields['dealer_name']} (conf: {confidence['dealer_name']:.2f})")
        print(f"  Model Name:     {fields['model_name']} (conf: {confidence['model_name']:.2f})")
        print(f"  Horse Power:    {fields['horse_power']} HP (conf: {confidence['horse_power']:.2f})")
        print(f"  Asset Cost:     ₹{fields['asset_cost']} (conf: {confidence['asset_cost']:.2f})")
        
        print(f"\n🖼️  VISUAL ELEMENTS:")
        print(f"  Signature:      {'✅ Present' if fields['signature']['present'] else '❌ Not Found'} (conf: {confidence['signature']:.2f})")
        if fields['signature']['present']:
            print(f"    Bbox:         {fields['signature']['bbox']}")
        
        print(f"  Stamp:          {'✅ Present' if fields['stamp']['present'] else '❌ Not Found'} (conf: {confidence['stamp']:.2f})")
        if fields['stamp']['present']:
            print(f"    Bbox:         {fields['stamp']['bbox']}")
        
        print(f"\n📈 METRICS:")
        print(f"  Overall Confidence: {results['overall_confidence']:.1%}")
        print(f"  Processing Time:    {results['processing_time_sec']}s")
        print(f"  Estimated Cost:     ${results['cost_estimate_usd']}")
        print("="*60 + "\n")


def main():
    """Main entry point with enhanced options"""
    parser = argparse.ArgumentParser(
        description="Enhanced Invoice Field Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF with standard quality
  python executable.py --pdf invoice.pdf
  
  # Process with aggressive preprocessing (for poor quality scans)
  python executable.py --pdf invoice.pdf --quality aggressive
  
  # Batch process directory
  python executable.py --batch ./invoices --quality standard
        """
    )
    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to single PDF file'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Path to directory containing PDF files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '--quality',
        type=str,
        choices=['fast', 'standard', 'aggressive'],
        default='standard',
        help='Preprocessing quality level (default: standard)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InvoiceExtractionPipeline(
        output_dir=args.output_dir,
        preprocessing_quality=args.quality
    )
    
    # Process based on input
    if args.pdf:
        # Single PDF
        result = pipeline.process_pdf(args.pdf)
        
        # Save result
        output_file = os.path.join(
            args.output_dir,
            f"{Path(args.pdf).stem}_result.json"
        )
        save_json_output(result, output_file)
        
        if result.get('success', False):
            print(f"✅ Results saved to: {output_file}\n")
        else:
            print(f"❌ Processing failed. Error details saved to: {output_file}\n")
    
    elif args.batch:
        # Batch processing
        pipeline.process_batch(args.batch)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
