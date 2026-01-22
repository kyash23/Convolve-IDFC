# app.py
"""
Gradio Web Interface for Invoice Extraction
Run: python app.py
Access: http://localhost:7860
"""
import gradio as gr
import os
import json
from pathlib import Path
from executable import InvoiceExtractionPipeline

# Initialize pipeline
pipeline = InvoiceExtractionPipeline(output_dir="output")

def extract_from_pdf(pdf_file):
    """
    Extract fields from uploaded PDF
    
    Args:
        pdf_file: Uploaded PDF file object from Gradio
    
    Returns:
        Formatted output string and JSON results
    """
    try:
        if pdf_file is None:
            return "❌ No PDF file uploaded", None
        
        # Get file path (Gradio uploads to temp directory)
        pdf_path = pdf_file
        
        # Process PDF
        result = pipeline.process_pdf(pdf_path)
        
        if "error" in result:
            return f"❌ Error: {result['error']}", None
        
        # Format output
        fields = result['fields']
        confidence = result['confidence']
        
        output_text = f"""
📄 **EXTRACTION RESULTS**

**TEXT FIELDS:**
- Dealer Name: {fields['dealer_name']} 
  Confidence: {confidence['dealer_name']:.1%}

- Model Name: {fields['model_name']} 
  Confidence: {confidence['model_name']:.1%}

- Horse Power: {fields['horse_power']} HP 
  Confidence: {confidence['horse_power']:.1%}

- Asset Cost: ₹{fields['asset_cost']} 
  Confidence: {confidence['asset_cost']:.1%}

**VISUAL ELEMENTS:**
- Signature: {'✓ Detected' if fields['signature']['present'] else '✗ Not Found'}
  Confidence: {confidence['signature']:.1%}
  Bbox: {fields['signature']['bbox']}

- Stamp: {'✓ Detected' if fields['stamp']['present'] else '✗ Not Found'}
  Confidence: {confidence['stamp']:.1%}
  Bbox: {fields['stamp']['bbox']}

**METRICS:**
- Overall Confidence: {result['overall_confidence']:.1%}
- Processing Time: {result['processing_time_sec']:.2f}s
- Estimated Cost: ${result['cost_estimate_usd']:.5f}
        """
        
        # Return formatted text and JSON
        json_output = json.dumps(result, indent=2)
        
        return output_text.strip(), json_output
    
    except Exception as e:
        error_msg = f"❌ Error processing PDF: {str(e)}"
        return error_msg, None

def clear_outputs():
    """Clear all outputs"""
    return None, None, None

# Create Gradio interface
with gr.Blocks(title="Invoice Field Extraction", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 Invoice Field Extraction System
    
    Extract key fields from tractor loan invoices using AI
    
    **Supported Fields:**
    - Dealer Name (fuzzy matching)
    - Model Name (pattern matching)
    - Horse Power (numeric extraction)
    - Asset Cost (numeric extraction)
    - Signature Detection (with bbox)
    - Stamp Detection (with bbox)
    """)
    
    with gr.Row():
        with gr.Column():
            # Input
            pdf_input = gr.File(
                label="Upload PDF Invoice",
                file_types=[".pdf"],
                type="filepath"
            )
            
            with gr.Row():
                extract_btn = gr.Button("🚀 Extract Fields", variant="primary")
                clear_btn = gr.Button("🔄 Clear", variant="secondary")
        
        with gr.Column():
            # Processing info
            gr.Markdown("""
            ### ℹ️ How It Works
            
            1. **Upload** your PDF invoice
            2. **Click Extract** to process
            3. **View results** with confidence scores
            4. **Download** JSON for integration
            
            **Processing takes 2-8 seconds**
            """)
    
    # Outputs
    with gr.Row():
        with gr.Column():
            output_text = gr.Markdown(label="📊 Results")
        
        with gr.Column():
            output_json = gr.Code(
                label="📋 JSON Output",
                language="json"
            )
    
    # Event handlers
    extract_btn.click(
        fn=extract_from_pdf,
        inputs=pdf_input,
        outputs=[output_text, output_json]
    )
    
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[pdf_input, output_text, output_json]
    )
    
    # Example usage
    gr.Examples(
        examples=[],  # Add example PDF paths here if available
        inputs=pdf_input,
        label="Examples (if available)"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True
    )
