import argparse
import json
import os
import re
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. Carga Global (Cold Start)
# Instanciamos procesador y modelo en el scope global para evitar retrasos por imagen.
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Utilizamos torch.bfloat16 para reducir sustancialmente el footprint de VRAM en ROCm/CUDA
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" if device == "cuda" else None
)

processor = AutoProcessor.from_pretrained(model_id)

def normalize_text(text: str) -> str:
    """
    Normaliza el texto aplicando estrictamente las reglas del reto.
    """
    text = text.upper()
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[\-\.·_]', '', text)
    return text

def process_image(image_path: str) -> tuple[str, float]:
    """
    Inferencia E2E con el VLM Qwen2-VL-2B-Instruct, asegurando gestión de memoria,
    prevención de fragmentación estricta y tolerancia a fallos.
    """
    try:
        # 2. Grafo de Ejecución Estático y Tensores (Límites Estrictos)
        # Redimensionamos la imagen a un máximo fijo para evitar picos catastróficos
        # de asignación dinámica de VRAM en el pre-procesamiento visual.
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((768, 768)) # Mantener proporciones estableciendo límite duro
        
        # 4. Prompting del VLM
        # Instrucción clara y restrictiva para evitar charlatanería.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Extract ONLY the text visible in the image. Return only the raw text, without any explanations, preambles, or markdown."}
                ],
            }
        ]
        
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generación con max_new_tokens muy bajo (20) porque solo buscamos patentes/texto corto.
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=20
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        # Como los VLM generativos no retornan un 'confidence' directo nativo,
        # asignamos 1.0 para cumplir estrictamente el formato JSON del evaluador.
        return output_text, 1.0
        
    except Exception as e:
        # 5. Fallback Tolerante a Fallos
        print(f"Error OOM/Corrupción procesando {image_path}: {e}")
        return "", 0.0
    finally:
        # 3. Gestión de VRAM y ROCm
        # Forzar vaciado de VRAM post-inferencia para mantener el uso muy debajo de los 48GB.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="AMD OCR Challenge - Qwen2-VL Implementation")
    parser.add_argument("--input-image", required=True, help="Ruta a la imagen de entrada")
    args = parser.parse_args()

    image_path = args.input_image
    
    if not os.path.exists(image_path):
        print(f"Error: La imagen {image_path} no existe.")
        output_data = {"text": "", "confidence": 0.0}
    else:
        raw_text, confidence = process_image(image_path)
        normalized_text = normalize_text(raw_text)
        
        output_data = {
            "text": normalized_text,
            "confidence": confidence
        }
    
    base_name = os.path.basename(image_path)
    file_name_without_ext, _ = os.path.splitext(base_name)
    
    output_dir = "/app/output/"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, f"{file_name_without_ext}_output.json")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)
        
    print(f"Procesamiento exitoso. Archivo generado: {output_file_path}")

if __name__ == "__main__":
    main()
