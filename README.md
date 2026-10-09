# AMD AI OCR Challenge - VLM Solution

This repository contains the solution for the AMD AI OCR Challenge. The objective of this project is to build an intelligent Optical Character Recognition (OCR) pipeline capable of reliably extracting text from uncooperative images, such as US and Chinese license plates, speed limits, and road work signs under adverse conditions (noise, blur, low light, and off-axis angles).

## Approach: Vision-Language Model (VLM)

Instead of relying on traditional OCR frameworks, this solution leverages a lightweight Vision-Language Model (VLM) to interpret the images contextually.

**Core Model:** Qwen2-VL-2B-Instruct.

**Why this model?** It provides a strong balance between visual comprehension (accurately reading complex characters like "京 A" on Chinese plates) and parameter efficiency, allowing it to easily fit within the strict hardware limits.

## Technologies

- **Frameworks:** PyTorch, Hugging Face Transformers.
- **Platform:** AMD ROCm (Radeon Open Compute).
- **Infrastructure:** Docker.

## Technical Constraints & Optimizations

The AMD evaluation environment imposes strict hardware and execution limits. This pipeline was specifically engineered to survive these constraints:

- **Memory Limit (48 GB VRAM):** The application strictly controls memory allocation. The model is loaded in `bfloat16` precision. We maintain a static execution graph by pre-allocating tensor shapes and limiting the `max_new_tokens` generation to prevent dynamic memory spikes. Explicit VRAM garbage collection (`torch.cuda.empty_cache()`) is triggered after every inference.
- **Execution Time (30s per image):** To meet the 30-second per-image inference limit, the VLM and its processor are loaded globally during a single "Cold Start" phase (which has a separate 10-minute budget).
- **Docker Environment:** The container is built strictly on top of the mandated `rocm/pytorch` base image. Uncompressed image size is kept well under the 60 GiB limit by handling dependencies via `requirements.txt` without squashing Docker layers, preserving the base image signature required by the automated grader.
- **Normalization Engine:** A robust Regex-based post-processing engine cleans the VLM output to exactly match the evaluation criteria (uppercase conversion, whitespace/punctuation stripping, and contextual filtering).

## Project Structure

- `app.py`: Main entry point handling CLI arguments, VLM initialization, inference, and E2E JSON formatting.
- `Dockerfile`: Container definition strictly adhering to the AMD evaluation runtime.
- `requirements.txt`: Python dependencies.

## How to Run (Runbook)

### 1. Build the Docker Image
```bash
docker build -t agustindiazcano/amd-ocr-challenge:v1 .
```

### 2. Run Local Inference
The application reads an input image and writes a JSON output file. Use volumes to map your local images and output directories to the container.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  -v $(pwd)/local_test_images:/app/input \
  -v $(pwd)/output:/app/output \
  agustindiazcano/amd-ocr-challenge:v1 \
  python app.py --input-image /app/input/sample.jpg
```
*(Note: Omit `--device=/dev/kfd --device=/dev/dri` if running CPU-only without AMD ROCm hardware).*

### 3. Output Format
The resulting JSON file will be created in your local `output/` folder (e.g., `sample_output.json`) following the exact challenge format:
```json
{"text": "京A12345", "confidence": 1.0}
```

## Author

**Agustin Diaz-Cano**  
*MS Candidate, Information Systems Engineering*
