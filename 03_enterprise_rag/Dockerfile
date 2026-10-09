FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Install system dependencies (Poppler for PDF handling, Tesseract for OCR if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install standard Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    requests \
    pdfplumber \
    python-docx \
    pandas \
    openpyxl \
    python-multipart \
    Pillow

# STRICT RULE: Install PyTorch-dependent libraries WITHOUT dependencies 
# to prevent pip from overwriting the ROCm build with CUDA versions.
RUN pip install --no-cache-dir --no-deps -U \
    transformers \
    qwen-vl-utils \
    accelerate \
    safetensors

# Copy application files
COPY daemon.py /app/
COPY app.py /app/

# The container must stay alive. We use Uvicorn as our long-running daemon.
# app.py will communicate with this daemon via HTTP.
CMD ["uvicorn", "daemon:app", "--host", "127.0.0.1", "--port", "8000"]
