# Convolve-IDFC
Below is a **clean, professional `README.md`** you can directly place in your project root.
It is written based on the **actual file structure and dependencies** you shared and is suitable for GitHub / academic submission.

---

# 📄 Multilingual Document OCR & Field Extraction System

This project is a **fully free, offline-capable OCR pipeline** for extracting structured information from **images and PDFs** containing multilingual text (e.g., English, Hindi, Gujarati, etc.).
It is designed to work **without paid APIs or subscriptions** and can run locally or on **Google Colab**.

The system supports:

* Multilingual OCR using PaddleOCR
* PDF and image input handling
* Field-wise text extraction
* Signature & stamp detection
* Modular, extensible architecture

---

## 📁 Project Structure

```text
.
├── app.py                         # Main application entry point
├── executable.py                  # CLI-style execution wrapper
├── run_images.py                  # Batch processing for image datasets
├── config.py                      # Central configuration (paths, flags, OCR settings)
│
├── ocr_engine.py                  # OCR logic using PaddleOCR
├── field_extractor.py             # Extracts structured fields from OCR text
├── signature_stamp_detector.py    # Detects signatures and stamps in documents
├── pdf_handler.py                 # PDF → image conversion and handling
├── utils.py                       # Helper and utility functions
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Features

* ✅ **Completely free & open-source**
* 🌐 **Multilingual OCR** (Indian + English languages)
* 🖼️ Handles **PNG / JPG images**
* 📄 Handles **PDF documents**
* ✍️ **Signature & stamp detection**
* 🧩 Modular design (easy to extend or replace components)
* 🧪 Suitable for **research, assignments, and production prototypes**

---

## 🧠 OCR Engine

The OCR functionality is powered by **PaddleOCR**, which:

* Supports multiple languages
* Works offline after installation
* Provides high accuracy for scanned documents

No API keys or cloud services are required.

---

## ⚙️ Installation

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`, including PaddleOCR, OpenCV, NumPy, and PDF utilities 

> ⚠️ **Note (PDF support)**
> `pdf2image` requires **Poppler**:
>
> **macOS**
>
> ```bash
> brew install poppler
> ```
>
> **Ubuntu**
>
> ```bash
> sudo apt install poppler-utils
> ```

---

## ▶️ How to Run

### 🔹 Run on a folder of images

```bash
python run_images.py
```

### 🔹 Run the main application

```bash
python app.py
```

### 🔹 CLI / executable-style run

```bash
python executable.py
```

Configuration options (input paths, OCR language, output formats) can be adjusted in:

```text
config.py
```

---

## 📤 Input

Supported inputs:

* Scanned document images (`.png`, `.jpg`)
* PDF files with scanned or embedded text

You can place your files in the configured input directory or pass paths programmatically.

---

## 📥 Output

The system produces:

* Extracted raw OCR text
* Structured field-level outputs (as dictionaries / JSON-like data)
* Detected signature and stamp regions
* Processed document results per file

Output format depends on your configuration and usage (can be easily extended).

---

## 🧩 Module Responsibilities

| File                          | Purpose                                |
| ----------------------------- | -------------------------------------- |
| `ocr_engine.py`               | Runs OCR on images                     |
| `pdf_handler.py`              | Converts PDFs to images                |
| `field_extractor.py`          | Extracts required fields from OCR text |
| `signature_stamp_detector.py` | Detects visual signatures & stamps     |
| `utils.py`                    | Shared helper functions                |
| `config.py`                   | Central configuration                  |

---

## 🧪 Platform Compatibility

* ✅ macOS (Intel & Apple Silicon)
* ✅ Linux
* ✅ Google Colab
* ⚠️ Windows (Poppler setup required for PDFs)

---

## 📌 Use Cases

* Academic assignments
* Government / form digitization
* Multilingual document processing
* OCR research projects
* Preprocessing for NLP pipelines

---

## 📜 License

This project uses **only free and open-source libraries**.
You are free to use, modify, and distribute it for academic or commercial purposes.

---

If you want, I can also:

* Generate a **sample output format**
* Add **example commands**
* Convert this into a **GitHub-ready README with badges**
* Or simplify it for **college assignment submission**

Just tell me 👍
