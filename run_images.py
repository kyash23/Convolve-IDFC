# run_images.py - Batch Image Processing
# Process all PNG/JPG images from a folder using the enhanced pipeline

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

from pdf_handler import ImagePreprocessor
from ocr_engine import OCREngine, TextProcessor
from field_extractor import FieldExtractor
from signature_stamp_detector import SignatureStampDetector, VisualElementValidator
from utils import save_json_output


def process_image(image_path: Path, ocr, preprocessor, extractor, sig_detector, verbose=False):
    """Process a single image and extract fields"""
    
    try:
        # Load image
        pil_img = Image.open(image_path).convert("RGB")
        image_array = np.array(pil_img)
        
        # Preprocess with standard quality
        processed = preprocessor.preprocess_pipeline(image_array, quality='standard')
        
        # OCR extraction
        ocr_results = ocr.extract_text(processed)
        if not ocr_results["success"]:
            if verbose:
                print(f"  ❌ OCR failed: {ocr_results.get('error', 'Unknown error')}")
            return None
        
        full_text = ocr_results["full_text"]
        detailed = ocr_results["detailed_results"]
        
        # Field extraction
        field_results = extractor.extract_all_fields(full_text, detailed)
        
        # Signature/stamp detection
        visual = sig_detector.detect_both(image_array)
        validator = VisualElementValidator()
        for key in ["signature", "stamp"]:
            if visual[key]["present"]:
                if not validator.validate_bbox(visual[key]["bbox"], image_array.shape):
                    visual[key]["present"] = False
                    visual[key]["confidence"] = 0.0
        
        # Calculate overall confidence (average of all fields)
        confidences = [
            field_results["dealer_name"]["confidence"],
            field_results["model_name"]["confidence"],
            field_results["horse_power"]["confidence"],
            field_results["asset_cost"]["confidence"],
            visual["signature"]["confidence"],
            visual["stamp"]["confidence"],
        ]
        overall_confidence = sum(confidences) / len(confidences)
        
        # Build result in required format
        result = {
            "doc_id": image_path.stem,
            "fields": {
                "dealer_name": field_results["dealer_name"]["value"],
                "model_name": field_results["model_name"]["value"],
                "horse_power": field_results["horse_power"]["value"],
                "asset_cost": field_results["asset_cost"]["value"],
                "signature": {
                    "present": visual["signature"]["present"],
                    "bbox": visual["signature"]["bbox"],
                },
                "stamp": {
                    "present": visual["stamp"]["present"],
                    "bbox": visual["stamp"]["bbox"],
                },
            },
            "confidence": round(overall_confidence, 3),
            "processing_time_sec": 0,  # Will be updated
            "cost_estimate_usd": 0,    # Will be updated
        }
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Error processing {image_path.name}: {str(e)}")
        return None


def main(path_str: str, quality: str = "standard", verbose: bool = False):
    """
    Main batch processing function
    
    Args:
        path_str: Path to image file or folder
        quality: 'fast', 'standard', or 'aggressive'
        verbose: Print detailed logs
    """
    
    path = Path(path_str)
    if not path.exists():
        print(f"❌ Path not found: {path}")
        return
    
    # Output directory
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("\n" + "="*70)
    print("ENHANCED INVOICE EXTRACTION PIPELINE - BATCH IMAGE PROCESSING")
    print("="*70)
    
    print("\n🔄 Initializing components...")
    ocr = OCREngine(languages=['en'], use_gpu=False)
    preprocessor = ImagePreprocessor()
    extractor = FieldExtractor()
    sig_detector = SignatureStampDetector()
    
    # Find images
    if path.is_file():
        img_paths = [path]
    else:
        img_paths = sorted([
            p for p in path.glob("*") 
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])
    
    print(f"✅ Found {len(img_paths)} image(s)\n")
    
    if len(img_paths) == 0:
        print("❌ No images found in the specified path")
        return
    
    # Process images
    results = []
    successful = 0
    failed = 0
    total_time = 0
    
    print("Processing images:")
    print("-" * 70)
    
    for i, img_path in enumerate(img_paths, 1):
        start = time.time()
        proc_time = time.time() - start
        
        result = process_image(img_path, ocr, preprocessor, extractor, sig_detector, verbose=verbose)
        
        proc_time = time.time() - start
        cost = proc_time * 0.001
        
        if result is not None:
            result["processing_time_sec"] = round(proc_time, 2)
            result["cost_estimate_usd"] = round(cost, 3)
            results.append(result)
            successful += 1
            total_time += proc_time
            
            # Print progress
            print(f"[{i:4d}/{len(img_paths)}] ✅ {img_path.name:40s} | conf: {result['confidence']:.3f} | time: {proc_time:5.1f}s")
            
            # Save individual result
            out_path = out_dir / f"{img_path.stem}_result.json"
            save_json_output(result, str(out_path))
        else:
            failed += 1
            print(f"[{i:4d}/{len(img_paths)}] ❌ {img_path.name:40s} | FAILED")
    
    print("-" * 70)
    
    # Save combined results
    if results:
        combined_path = out_dir / "all_results.json"
        save_json_output(results, str(combined_path))
        
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_time = total_time / successful if successful > 0 else 0
        
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"✅ Successfully processed: {successful}/{len(img_paths)}")
        print(f"❌ Failed: {failed}/{len(img_paths)}")
        print(f"📊 Average confidence: {avg_confidence:.3f}")
        print(f"⏱️  Average time per image: {avg_time:.2f}s")
        print(f"💾 Total time: {total_time:.1f}s")
        print(f"📁 Results saved to: {out_dir}/")
        print(f"   - Individual: {img_path.stem}_result.json")
        print(f"   - Combined: all_results.json")
        print("="*70 + "\n")
    else:
        print("\n❌ No images were successfully processed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_images.py /path/to/image.png")
        print("  python run_images.py /path/to/folder")
        print("  python run_images.py /path/to/folder --quality aggressive")
        print("  python run_images.py /path/to/folder --verbose")
        sys.exit(1)
    
    # Parse arguments
    path = sys.argv[1]
    quality = "standard"
    verbose = False
    
    for arg in sys.argv[2:]:
        if arg.startswith("--quality"):
            quality = arg.split("=")[1] if "=" in arg else "standard"
        elif arg == "--verbose":
            verbose = True
    
    main(path, quality=quality, verbose=verbose)
