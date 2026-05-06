import os
import io
import json
import time
import re
import fitz  # PyMuPDF
import torch
import logging
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from contextlib import asynccontextmanager
from typing import Optional

# VLM Imports
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Setup proper logging for Docker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

# ==========================================
# 🧠 2. OCR ENGINE CLASS
# ==========================================

class QwenOCREngine:
    """Hugging Face Qwen3.5 Implementation"""
    def __init__(self, config):
        logger.info("Initializing Qwen OCR Engine...")
        self.model_path = config["model_path"]
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        # self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.dtype = torch.float32
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        # ✅ Adopted ChatGPT's advice: Removed device_map="auto" and used .to(device)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model.eval()
        
        # ❌ Ignored ChatGPT's advice here: Keeping 2048 for Arabic PDF safety
        self.max_new_tokens = config.get("max_new_tokens", 2048) 
        self.repetition_penalty = config.get("repetition_penalty", 1.2)
        logger.info(f"✅ Qwen Engine loaded successfully on {self.device}")

    def extract_text(self, image: Image.Image, prompt: str) -> str:
        # Resize to multiples of 64
        w, h = image.size
        new_w = ((w + 63) // 64) * 64
        new_h = ((h + 63) // 64) * 64
        if (new_w, new_h) != (w, h):
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=3,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        input_len = inputs.input_ids.shape[1]
        output_text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()

# ==========================================
#  3. HELPER FUNCTIONS
# ==========================================

def clean_output(text: str, max_repetitions: int = 2) -> str:
    if not text or not CONFIG["api_settings"].get("clean_output_repetitions", True):
        return text
    
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    lines = text.strip().split('\n')
    cleaned, seen = [], {}
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        count = seen.get(line_stripped, 0) + 1
        if count <= max_repetitions:
            cleaned.append(line)
        seen[line_stripped] = count
            
    return '\n'.join(cleaned).strip()

def pdf_page_to_image(pdf_bytes: bytes, page_num: int, dpi: int) -> Image.Image:
    doc = fitz.open("pdf", pdf_bytes)
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

# ==========================================
# 🚀 4. FASTAPI LIFESPAN & ROUTING
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Load the model
    logger.info("Starting up API and loading AI Model into memory...")
    engine_config = CONFIG.get("engines", {}).get("transformers", CONFIG)
    app.state.engine = QwenOCREngine(engine_config)
    yield
    # Shutdown logic: Clean up memory
    logger.info("Shutting down API and clearing memory...")
    del app.state.engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="Arabic OCR Microservice", version="1.0", lifespan=lifespan)

@app.get("/health")
def health_check(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "engine"),
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/ocr/image")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        final_prompt = prompt or CONFIG["api_settings"]["default_prompt"]
        
        start_time = time.time()
        # Access the engine safely from the app state
        raw_text = request.app.state.engine.extract_text(image, final_prompt)
        cleaned_text = clean_output(raw_text)
        elapsed = round(time.time() - start_time, 2)
        
        return {
            "filename": file.filename,
            "text": cleaned_text,
            "time_seconds": elapsed
        }
    except Exception as e:
        logger.error(f"Image processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/pdf")
async def process_pdf(
    request: Request,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    start_page: Optional[int] = Form(0),
    end_page: Optional[int] = Form(None)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        pdf_bytes = await file.read()
        doc = fitz.open("pdf", pdf_bytes)
        total_pages = len(doc)
        doc.close()
        
        final_prompt = prompt or CONFIG["api_settings"]["default_prompt"]
        dpi = CONFIG["engines"]["transformers"].get("pdf_dpi", 150)
        
        actual_end = end_page if end_page is not None else total_pages
        actual_end = min(actual_end, total_pages)
        
        results = []
        total_start = time.time()
        
        for page_num in range(start_page, actual_end):
            page_start = time.time()
            
            image = pdf_page_to_image(pdf_bytes, page_num, dpi)
            # Access the engine safely from the app state
            raw_text = request.app.state.engine.extract_text(image, final_prompt)
            cleaned_text = clean_output(raw_text)
            
            results.append({
                "page": page_num + 1,
                "text": cleaned_text,
                "time_seconds": round(time.time() - page_start, 2)
            })
            
        total_time = round(time.time() - total_start, 2)
        
        return {
            "filename": file.filename,
            "total_pages_processed": len(results),
            "total_time_seconds": total_time,
            "pages": results
        }
    except Exception as e:
        logger.error(f"PDF processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clear VRAM after processing a heavy PDF
        if torch.cuda.is_available():
            torch.cuda.empty_cache()