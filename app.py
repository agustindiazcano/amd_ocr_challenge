import argparse
import json
import os
import re

def normalize_text(text: str) -> str:
    """
    Normalizes the text according to the specific challenge rules:
    - Convert to uppercase
    - Remove all whitespaces
    - Remove characters: -, ., ·, _
    """
    # Convert to uppercase
    text = text.upper()
    # Remove all whitespaces
    text = re.sub(r'\s+', '', text)
    # Remove exact characters: -, ., ·, _
    text = re.sub(r'[\-\.·_]', '', text)
    return text

def process_image_mock(image_path: str):
    """
    Mocks the vision model inference.
    Returns a dirty string and a confidence score.
    """
    return "  京 A·123-45_ ", 0.95

def main():
    parser = argparse.ArgumentParser(description="AMD OCR Challenge - E2E Testing")
    parser.add_argument("--input-image", required=True, help="Path to the input image")
    args = parser.parse_args()

    image_path = args.input_image
    
    # Note: Pillow or other libraries could be used here to read the image
    # For now, we use the mock
    
    raw_text, confidence = process_image_mock(image_path)
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
        
    print(f"Success. Wrote output to {output_file_path}")

if __name__ == "__main__":
    main()
