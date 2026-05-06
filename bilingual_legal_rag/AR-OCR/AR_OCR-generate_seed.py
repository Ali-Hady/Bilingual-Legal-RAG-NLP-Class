import os
import json
import fitz  # PyMuPDF
import torch
import logging
from pathlib import Path
from PIL import Image

# Import the core logic from your existing microservice
from ocrModel import QwenOCREngine, CONFIG, clean_output

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Dynamically resolve paths so it never breaks
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIRECTORY = BASE_DIR / "../../seed_data/raw_arabic"
OUTPUT_JSON = BASE_DIR / "../../seed_data/arabic_laws_seed.json"

def generate_seed_data(input_folder: Path, output_json_path: Path):
    logger.info(f"🚀 Starting Seed Generation...")
    logger.info(f"📂 Input Folder: {input_folder}")
    logger.info(f"💾 Output File: {output_json_path}")
    
    if not input_folder.exists():
        logger.error(f"Folder {input_folder} does not exist!")
        return

    # 1. Initialize Engine
    engine_config = CONFIG.get("engines", {}).get("transformers", CONFIG)
    engine = QwenOCREngine(engine_config)
    prompt = CONFIG["api_settings"]["default_prompt"]
    dpi = engine_config.get("pdf_dpi", 150)
    max_image_size = 2000 # Safeguard against massive scanned documents
    
    results = []
    valid_image_exts = {".png", ".jpg", ".jpeg"}
    
    # 2. Iterate through files
    for file_path in input_folder.rglob("*"):
        if not file_path.is_file():
            continue
            
        ext = file_path.suffix.lower()
        
        try:
            # --- HANDLE IMAGES ---
            if ext in valid_image_exts:
                logger.info(f"🖼️ Processing Image: {file_path.name}")
                image = Image.open(file_path).convert("RGB")
                
                # Safeguard: Resize massive images before OCR
                if max(image.size) > max_image_size:
                    image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
                
                raw_text = engine.extract_text(image, prompt)
                cleaned = clean_output(raw_text)
                
                if cleaned.strip():
                    results.append({
                        "source_document": file_path.name,
                        "document_type": "image",
                        "page_number": 1,
                        "text": cleaned,
                        "file_path": str(file_path)
                    })
                
            # --- HANDLE PDFs ---
            elif ext == ".pdf":
                logger.info(f"📄 Processing PDF: {file_path.name}")
                doc = fitz.open(file_path) # Open once!
                total_pages = len(doc)
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                
                for page_num in range(total_pages):
                    logger.info(f"   -> Extracting Page {page_num + 1}/{total_pages}...")
                    
                    # Convert page to image inline to avoid reopening the PDF
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=mat)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    if max(image.size) > max_image_size:
                        image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
                    
                    raw_text = engine.extract_text(image, prompt)
                    cleaned = clean_output(raw_text)
                    
                    if cleaned.strip():
                        results.append({
                            "source_document": file_path.name,
                            "document_type": "pdf",
                            "page_number": page_num + 1,
                            "text": cleaned,
                            "file_path": str(file_path)
                        })
                    
                    # Periodic VRAM cleanup for stability
                    if torch.cuda.is_available() and page_num % 5 == 0:
                        torch.cuda.empty_cache()
                        
                    # Incremental Save every 10 pages to prevent data loss
                    if len(results) % 10 == 0:
                        output_json_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_json_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)

                doc.close()
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}", exc_info=True)

    # 3. Final Save
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"\n[SUCCESS] ✅ Seed data saved to {output_json_path}")
    logger.info(f"[STAT] Total valid text chunks generated: {len(results)}")

if __name__ == "__main__":
    generate_seed_data(INPUT_DIRECTORY, OUTPUT_JSON)