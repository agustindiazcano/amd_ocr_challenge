# AMD AI Challenge - Monorepo

This repository contains two independent projects developed for the AMD AI Challenge. Both architectures are heavily optimized to run efficiently on AMD ROCm hardware under strict constraints: a maximum of **48 GB VRAM** and **<30 seconds of execution time** per inference.

## Table of Contents
1. [02: VLM OCR Pipeline](#02-vlm-ocr-pipeline)
2. [03: Enterprise RAG Engine](#03-enterprise-rag-engine)
3. [04: Intelligent Web Information Retrieval](#04-intelligent-web-information-retrieval)
4. [05: AI-Powered Code Repository Repair](#05-ai-powered-code-repository-repair)
5. [06: Can Your AI Master the Unknown](#06-can-your-ai-master-the-unknown)
6. [Author](#author)

---

## 02: VLM OCR Pipeline

**Folder:** `02_vlm_ocr/`

The objective of this project is to build an intelligent Optical Character Recognition (OCR) pipeline capable of reliably extracting text from uncooperative images, such as US and Chinese license plates, speed limits, and road work signs under adverse conditions (noise, blur, low light, and off-axis angles).

### Approach: Vision-Language Model (VLM)
Instead of relying on traditional OCR frameworks, this solution leverages a lightweight Vision-Language Model (VLM) to interpret the images contextually.

**Core Model:** Qwen2-VL-2B-Instruct.

**Why this model?** It provides a strong balance between visual comprehension (accurately reading complex characters like "京 A" on Chinese plates) and parameter efficiency, allowing it to easily fit within the strict hardware limits.

### Technologies
- **Frameworks:** PyTorch, Hugging Face Transformers.
- **Platform:** AMD ROCm (Radeon Open Compute).
- **Infrastructure:** Docker.

### Technical Constraints & Optimizations
The AMD evaluation environment imposes strict hardware and execution limits. This pipeline was specifically engineered to survive these constraints:
- **Memory Limit (48 GB VRAM):** The application strictly controls memory allocation. The model is loaded in `bfloat16` precision. We maintain a static execution graph by pre-allocating tensor shapes and limiting the `max_new_tokens` generation to prevent dynamic memory spikes. Explicit VRAM garbage collection (`torch.cuda.empty_cache()`) is triggered after every inference.
- **Execution Time (30s per image):** To meet the 30-second per-image inference limit, the VLM and its processor are loaded globally during a single "Cold Start" phase (which has a separate 10-minute budget).
- **Docker Environment:** The container is built strictly on top of the mandated `rocm/pytorch` base image. Uncompressed image size is kept well under the 60 GiB limit by handling dependencies via `requirements.txt` without squashing Docker layers, preserving the base image signature required by the automated grader.
- **Normalization Engine:** A robust Regex-based post-processing engine cleans the VLM output to exactly match the evaluation criteria (uppercase conversion, whitespace/punctuation stripping, and contextual filtering).

### How to Run (Runbook)
**1. Build the Docker Image**
```bash
cd 02_vlm_ocr
docker build -t agustindiazcano/amd-ocr-challenge:v1 .
```

**2. Run Local Inference**
```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  -v $(pwd)/local_test_images:/app/input \
  -v $(pwd)/output:/app/output \
  agustindiazcano/amd-ocr-challenge:v1 \
  python app.py --input-image /app/input/sample.jpg
```
*(Note: Omit `--device=/dev/kfd --device=/dev/dri` if running CPU-only without AMD ROCm hardware).*

---

## 03: Enterprise RAG Engine

**Folder:** `03_enterprise_rag/`

The objective of this project is to build an Enterprise Retrieval-Augmented Generation (RAG) engine capable of ingesting a mixed corpus of internal documents (.pdf, .docx, .xlsx, .csv, .txt, .log, .py, .png, .jpg) and answering extremely precise queries under a strict 30-second inference limit per question.

### Approach: Client-Server Architecture + Deterministic RAG
Because the evaluator executes the `app.py` script repeatedly as a completely new process for each query, loading the model at every execution would immediately violate the 30-second limit. We solve this by implementing a **Client-Server architecture**:
- **Daemon (`daemon.py`)**: A persistent FastAPI server runs in the background holding the VLM (`Qwen2-VL-2B-Instruct`) and a local `SQLite FTS5` index in memory.
- **Thin Client (`app.py`)**: A lightweight CLI script that passes the evaluator's arguments directly to the daemon via HTTP and reliably writes the JSON output.

### Technologies
- **Frameworks**: FastAPI, PyTorch, Hugging Face Transformers, SQLite FTS5, pdfplumber, openpyxl, python-docx.
- **Platform**: AMD ROCm (Radeon Open Compute).
- **Infrastructure**: Docker.

### Technical Constraints & Optimizations
- **Memory Limit (48 GB VRAM)**: We enforce strict VRAM pre-allocation with fixed array shapes (`max_length=2048`) to maintain a static execution graph, eliminating dynamic memory spikes. Explicit garbage collection (`torch.cuda.empty_cache()`) clears any visual tensors post-inference.
- **Causal Citations**: The system implements a strict causal necessity rule for citations. A document is only cited if removing it makes the answer impossible. The FTS5 index relies purely on deterministic lexical search (BM25) over abstract vectors to ensure no "hallucinated" citations.
- **Fault-Tolerance (Corpus Traps)**: The indexing engine is fortified against deliberate evaluator traps. It silently logs and bypasses completely empty directories, unrecognized binary formats, files lacking read permissions (`chmod 000` dropped via `DAC_OVERRIDE`), and password-protected encrypted PDFs, without crashing or halting the pipeline.
- **Strict Evaluator Dependencies**: Deep learning packages (`transformers`, `qwen-vl-utils`) are strictly installed using `--no-deps` to prevent standard `pip` from silently overwriting the mandated ROCm PyTorch base with an incompatible CUDA build.

### How to Run (Runbook)
**1. Build the Docker Image**
```bash
cd 03_enterprise_rag
docker build -t agustindiazcano/amd-rag-challenge:v1 .
```

**2. Run Local Evaluation Simulation**
You can run the simulated hostile environment directly:
```bash
cd 03_enterprise_rag
python test_traps.py
```
This script will assemble a hostile corpus with unreadable files, start the daemon, perform the 10-minute indexing pass, run an isolated query passing the 30-second strict limit, and evaluate the fallback resilience.

---

## 04: Intelligent Web Information Retrieval

**Folder:** `04_intelligent_web_information_retrieval/`

*(Upcoming)* Details for the fourth phase of the challenge will be placed here.

---

## 05: AI-Powered Code Repository Repair

**Folder:** `05_ai_powered_code_repository_repair/`

*(Upcoming)* Details for the fifth phase of the challenge will be placed here.

---

## 06: Can Your AI Master the Unknown

**Folder:** `06_can_your_ai_master_the_unknown/`

*(Upcoming)* Details for the sixth and final phase of the challenge will be placed here.

---

## Author

**Agustin Diaz-Cano**  
*MS Candidate, Information Systems Engineering*


---
*These projects are official submissions for the [AMD & LabLab AI Academy Challenge](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge).*
