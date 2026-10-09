import os
import logging
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import torch

from parsers import CorpusIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Pydantic Models ---
class IndexRequest(BaseModel):
    corpus_path: str

class QueryRequest(BaseModel):
    query_id: str
    query: str
    corpus_path: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float

# --- Global State / Model Loader ---
class VLMContext:
    def __init__(self):
        self.model = None
        self.processor = None
        
    def initialize(self):
        """Loads the VLM model into VRAM enforcing static pre-allocation."""
        logger.info("Initializing VLM model...")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model_id = "Qwen/Qwen2-VL-2B-Instruct"
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda", # Automatically mapped to ROCm
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            self._warmup_static_shapes()
        except ImportError:
            logger.warning("Transformers not found (Running locally?). Skipping VLM load.")

    def _warmup_static_shapes(self):
        logger.info("Performing warmup with static fixed shapes (padding='max_length') to pre-allocate VRAM...")
        try:
            dummy_input = "System warmup"
            inputs = self.processor(
                text=[dummy_input], 
                return_tensors="pt", 
                padding="max_length", 
                max_length=2048,
                truncation=True
            ).to("cuda")
            
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=10)
            logger.info("Static pre-allocation warmup completed.")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def extract_text_from_image(self, image_path: str) -> str:
        """Runs the VLM specifically for OCR and diagram descriptions on images."""
        try:
            from qwen_vl_utils import process_vision_info
            
            prompt = "Extract all visible text exactly as written. Describe any diagrams, schemas, pinouts, labels, or tables in detail."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Static shape padding
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Strict memory rule: Clear explicit tensors from VRAM to prevent dynamic fragmentation
            del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
            torch.cuda.empty_cache()
            
            return output_text
        except Exception as e:
            logger.error(f"VLM inference failed on image {image_path}: {e}")
            return ""

    def generate_rag_answer(self, prompt: str) -> str:
        """Infects exact strict RAG instructions to extract pure values deterministically."""
        try:
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a rigorous data extraction engine. You will be provided with a set of documents and a question. "
                        "Your task is to find the EXACT value answering the question.\n"
                        "RULES:\n"
                        "1. Output ONLY a valid JSON object.\n"
                        "2. The JSON must have exactly two keys: 'answer' and 'citations'.\n"
                        "3. The 'answer' must be the exact value ONLY (e.g., '94', 'Q3 FY27'). NO prose, NO sentences.\n"
                        "4. If the answer is NOT explicitly found in the provided documents, the 'answer' MUST be \"\" and 'citations' MUST be []. Do NOT guess from prior knowledge.\n"
                        "5. The 'citations' MUST be a list containing the file_path of the document(s) used. ONLY cite documents strictly necessary to extract the answer."
                    )
                },
                {"role": "user", "content": prompt}
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Static shapes for memory stability
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to("cuda")

            with torch.no_grad():
                # temperature=0.0 forces deterministic output
                generated_ids = self.model.generate(**inputs, max_new_tokens=150, temperature=0.0)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()
            
            return output_text
        except Exception as e:
            logger.error(f"VLM RAG generation failed: {e}")
            return ""

context = VLMContext()

@app.on_event("startup")
async def startup_event():
    # Initialize the model right at the beginning
    context.initialize()
    
    # Initialize Corpus Indexer with local DB
    app.state.indexer = CorpusIndexer(db_path="/app/corpus_index.db", vlm_context=context)
    logger.info("Daemon started and listening.")

@app.post("/index")
async def index_corpus(request: IndexRequest):
    logger.info(f"Starting indexing for corpus: {request.corpus_path}")
    indexed_count = 0
    indexer = app.state.indexer
    
    if not os.path.exists(request.corpus_path):
        return {"status": "error", "message": "Corpus path does not exist"}

    for root, dirs, files in os.walk(request.corpus_path):
        # Gracefully handle empty directories
        if not files and not dirs:
            logger.debug(f"Skipping empty directory: {root}")
            continue

        for file in files:
            file_path = os.path.join(root, file)
            
            # Check for generic read permission
            if not os.access(file_path, os.R_OK):
                logger.warning(f"Skipping unreadable file (No Permission): {file_path}")
                continue
            
            try:
                # Attempt to open to catch 'chmod 000' applied under DAC_OVERRIDE drop
                with open(file_path, "rb") as f:
                    _ = f.read(10)
                
                # Pass to parsers
                # We save relative paths to match exactly what the user requested, 
                # or keep the absolute path from the crawl
                # According to the brief, both are accepted (e.g., /app/corpus/specs/x.pdf or specs/x.pdf)
                # We will keep the exact path provided by os.walk which starts with request.corpus_path
                # But it's usually cleaner to store relative paths inside the index if we want.
                # Let's just use the file_path as is.
                if indexer.index_file(file_path):
                    indexed_count += 1
                
            except PermissionError:
                logger.warning(f"Permission error (DAC_OVERRIDE dropped) reading file: {file_path}")
            except Exception as e:
                logger.warning(f"Error reading file {file_path}. Skipping. Reason: {str(e)}")

    logger.info(f"Indexing complete. Processed {indexed_count} readable files.")
    return {"status": "success", "indexed_files": indexed_count}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    import re
    import json
    logger.info(f"Processing query {request.query_id}: {request.query}")
    
    answer = ""
    citations = []
    confidence = 0.0

    indexer = app.state.indexer
    
    # 1. Deterministic FTS5 Retrieval
    retrieved_docs = indexer.search(request.query, top_k=3)
    
    if not retrieved_docs:
        logger.info("No relevant documents found. Bailing out early.")
        return QueryResponse(answer="", citations=[], confidence=0.0)
        
    # 2. Context Formulation
    prompt = "DOCUMENTS:\n"
    retrieved_paths = []
    for doc in retrieved_docs:
        # We strip the leading corpus path to try to make citations relative, 
        # or just use the exact path stored. Let's use the exact path stored.
        path = doc["file_path"]
        retrieved_paths.append(path)
        prompt += f"[FILE: {path}]\n{doc['content']}\n\n"
        
    prompt += f"QUESTION: {request.query}\n"
    
    # 3. Model Generation
    raw_output = context.generate_rag_answer(prompt)
    logger.info(f"Model raw output: {raw_output}")
    
    # 4. Deterministic Extraction via Regex
    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            
            # Enforce Causal Citation & Empty Rules
            parsed_answer = str(parsed.get("answer", "")).strip()
            parsed_citations = parsed.get("citations", [])
            
            if parsed_answer and parsed_answer.lower() not in ["none", "null", "n/a", "unknown"]:
                answer = parsed_answer
                # Only keep valid citations that were actually injected
                citations = [c for c in parsed_citations if c in retrieved_paths]
            else:
                answer = ""
                citations = []
                
        except json.JSONDecodeError:
            logger.error("JSON decode error on model output.")
            
    return QueryResponse(answer=answer, citations=citations, confidence=confidence)
