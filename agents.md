# AMD ROCm ML Pipelines - Project Context

## Overview
This monorepo contains end-to-end Machine Learning solutions optimized to run on AMD ROCm hardware. Originally designed for the AMD AI Challenge, it consists of specialized Vision-Language Model (VLM) architectures for OCR and Enterprise Retrieval-Augmented Generation (RAG). 

The primary focus of all code is rigorous engineering, hardware optimization, strict memory control, and deterministic execution. Physical mechanisms, structured logic, and static pre-allocation are prioritized over abstract or dynamic statistical models.

## Strict Infrastructure Constraints (System Rules)
- **Base Environment:** Mandatory Docker images based on `rocm/pytorch`. Maximum uncompressed image size limit: 60 GB.
- **VRAM Memory:** Strict 48 GB limit. Mandatory use of static pre-allocation for tensor shapes (fixed array shapes) to maintain a static execution graph, avoiding memory fragmentation and dynamic concatenation errors.
- **Execution Latency:** Strict 30-second limit per query/inference.
- **Cold Start & Indexing:** 10 minutes maximum to start the container, load models into VRAM, and index documents.

## Monorepo Structure and Current Status

### `/02_vlm_ocr` (Status: Completed)
- **Objective:** Robust OCR pipeline using Qwen2-VL-2B-Instruct for text extraction in hostile images.
- **Implemented Techniques:** `bfloat16` precision, VRAM control with static tensors, Regex post-processing engine to ensure a deterministic JSON output.

### `/03_enterprise_rag` (Status: In Development - Current Focus)
- **Objective:** Enterprise RAG engine capable of ingesting and answering queries on a mixed corpus of files (.pdf, .docx, .xlsx, .csv, .txt, .log, .py, .png, .jpg).
- **Mandatory Architecture:** Since the evaluator invokes `python app.py` as a completely new process for each question, loading the model in every iteration exhausts the 30-second limit. **A Client-Server architecture (daemon / UNIX socket / local API) must be implemented** where the model resides in a persistent background process and `app.py` acts as a thin client.
- **Citation Rule:** Strict causal necessity. A document is only cited if removing it makes it impossible to answer the question. Citing for mere "thematic relevance" ruins the score.
- **Fault Tolerance:** The indexer must bypass empty directories, files with `chmod 000` (without `DAC_OVERRIDE`), encrypted files, and unknown formats without throwing fatal exceptions or halting execution.

## Instructions for the Code Assistant
1. Write deterministic, modular, and strongly typed code.
2. Avoid unnecessary abstractions that obscure memory consumption.
3. Apply the principle of parsimony: if a model or code block explains/solves the problem with fewer variables and direct operations, it is the one closest to reality.
4. Keep dependency installations clean (use `--no-deps` in Dockerfiles when updating critical modules like Transformers to avoid overwriting the ROCm build with CUDA).

## Supported Agents
The RAG engine supports different LLM backends to power the conversational and retrieval capabilities. 
- [Gemini Implementation](gemini.md)
- [Claude Implementation](claude.md)
