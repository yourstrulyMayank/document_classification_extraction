# =====================================
# CONFIGURATION BLOCK
# =====================================

# --- LLM Provider Settings ---
MODEL_PROVIDER = "api"  # options: "huggingface", "ollama", "api"
MODEL_NAME = "google/gemma-3-1b-it"  # e.g., "gemma-7b-it", "llama3:8b", "mistral-7b-instruct" #Not required if provider is api
HF_LOCAL_PATH = r"/models/llm/gemma-3-1b-it"  # used if MODEL_PROVIDER = "huggingface"

# --- OCR Settings ---
OCR_ENGINE = "gemini"  # options: "easyocr", "paddleocr", "tesseract", "gemini"
OCR_LANGUAGES = ['en']

# --- Gemini API Settings ---
GEMINI_API_KEY = "AIzaSyC1UyxGaDx7j2caQz_F5XYy6-08rMYzJ8Q"  # or use env var
GEMINI_MODELS = {1: 'gemini-2.5-pro', 
                 2: 'gemini-2.5-flash',
                 3: 'gemini-2.5-flash-lite',
                 4: 'gemini-2.0-flash',
                 5: 'gemini-2.0-flash-lite',
                 6: 'gemini-2.0-flash-live-001'}
GEMINI_MODEL = GEMINI_MODELS[5]  


# --- Categories ---
CATEGORIES = [
    "passport", "driving_license", "national_id",
    "w8_certificate", "w9_certificate", "home_loan_documents"
]


# =====================================
# IMPORTS & INIT
# =====================================

from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for PDF handling
import re
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import base64
import io
import easyocr
import json
import subprocess
from pydantic import BaseModel
# import whisper
import librosa
import soundfile as sf
import PIL
# from moviepy.editor import VideoFileClip
import tempfile
from common_functions import (
    get_gliner_model, get_spacy_model, reader,
    identify_pii, redact_pii, draw_black_rectangles, gliner_model, spacy_model
)
# LLM Init
llm_model = None  # Initialize as None
if MODEL_PROVIDER == "huggingface":
    from transformers import pipeline
    if os.path.exists(HF_LOCAL_PATH):
        llm_pipe = pipeline("text-generation", model=HF_LOCAL_PATH, device_map="auto")
    else:
        llm_pipe = pipeline("text-generation", model=MODEL_NAME, device_map="auto")
    llm_model = llm_pipe  # Set the llm_model for common_functions
elif MODEL_PROVIDER == "ollama":
    # For Ollama, we'll create a wrapper class to match the interface expected by common_functions
    class OllamaWrapper:
        def __init__(self, model_name):
            self.model_name = model_name
        
        def invoke(self, prompt):
            try:
                result = subprocess.run(
                    ["ollama", "run", self.model_name],
                    input=prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                    encoding='utf-8',
                    errors='ignore'
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    print(f"Ollama error: {result.stderr}")
                    return ""
            except Exception as e:
                print(f"Ollama error: {e}")
                return ""
    
    llm_model = OllamaWrapper(MODEL_NAME)
elif MODEL_PROVIDER == "api":
    # For Gemini API, create wrapper class
    class GeminiWrapper:
        def __init__(self):
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
        
        def invoke(self, prompt):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini API error: {e}")
                return ""
    
    llm_model = GeminiWrapper()


# OCR Init


# =====================================
# CORE HELPERS
# =====================================

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Preprocess image for better OCR results"""
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Save preprocessed image temporarily
    temp_path = image_path.replace('.', '_processed.')
    cv2.imwrite(temp_path, thresh)
    
    return temp_path

def query_model(prompt):
    """Send prompt to selected LLM provider and return text."""
    if MODEL_PROVIDER == "huggingface":
        result = llm_pipe(prompt, max_new_tokens=512, do_sample=False)
        # Fix: Extract only the new generated text, not the entire prompt + response
        generated_text = result[0]['generated_text']
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text
    elif MODEL_PROVIDER == "ollama":
        try:
            result = subprocess.run(
                ["ollama", "run", MODEL_NAME],
                input=prompt,
                text=True,  # Add this for proper text handling
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60, # Add timeout to prevent hanging
                encoding='utf-8',  # Force UTF-8 encoding
                errors='ignore'    # Ignore decode errors
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Ollama error: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            print("Ollama request timed out")
            return ""
        except Exception as e:
            print(f"Ollama subprocess error: {e}")
            return ""
    elif MODEL_PROVIDER == "api":
        # Gemini API call
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")

def get_ocr_reader(ocr_engine, languages=['en']):
    """Initialize OCR reader based on selected engine"""
    if ocr_engine == "easyocr":
        import easyocr
        return easyocr.Reader(languages)
    elif ocr_engine == "paddleocr":
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='en')
    elif ocr_engine == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)
    else:  # tesseract
        return None

def extract_text_with_engine(image_path, ocr_engine):
    """Extract text using specified OCR engine"""
    try:
        processed_path = preprocess_image(image_path)
        print(f"Using OCR engine: {ocr_engine}")
        
        if ocr_engine == "easyocr":
            reader = get_ocr_reader(ocr_engine)
            result = reader.readtext(processed_path, detail=0, paragraph=True)
            text = "\n".join(result).strip()

        elif ocr_engine == "paddleocr":
            reader = get_ocr_reader(ocr_engine)
            result = reader.predict(processed_path, cls=True)
            text = "\n".join([line[1][0] for page in result for line in page]).strip()

        elif ocr_engine == "tesseract":
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(processed_path))

        elif ocr_engine == "gemini":
            import PIL.Image
            model = get_ocr_reader(ocr_engine)
            with PIL.Image.open(processed_path) as img:  # Use context manager
                response = model.generate_content(
                    ["Extract all readable text from this document image:", img]
                )
                text = response.text.strip()

        else:
            raise ValueError(f"Unsupported OCR engine: {ocr_engine}")

        # Safe file deletion with retry
        if os.path.exists(processed_path):
            try:
                import time
                time.sleep(0.1)  # Brief delay
                os.remove(processed_path)
            except PermissionError:
                print(f"Warning: Could not delete temporary file {processed_path}")

        return text

    except Exception as e:
        print(f"OCR error with {ocr_engine}: {e}")
        return ""
        
def extract_text_from_image(image_path):
    """Run OCR using the configured OCR engine."""
    if OCR_ENGINE == "easyocr":
        import easyocr
        ocr_reader = easyocr.Reader(OCR_LANGUAGES)

    elif OCR_ENGINE == "paddleocr":
        from paddleocr import PaddleOCR
        ocr_reader = PaddleOCR(use_textline_orientation=True, lang='en')

    elif OCR_ENGINE == "tesseract":
        import pytesseract
        from PIL import Image
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust path

    elif OCR_ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
    try:
        processed_path = preprocess_image(image_path)
        print(f"Processed image saved to: {processed_path}")
        text = ""
        
        if OCR_ENGINE == "easyocr":
            result = ocr_reader.readtext(processed_path, detail=0, paragraph=True)
            text = "\n".join(result).strip()

        elif OCR_ENGINE == "paddleocr":
            result = ocr_reader.predict(processed_path)
            text = "\n".join([line[1][0] for page in result for line in page]).strip()

        elif OCR_ENGINE == "tesseract":
            with Image.open(processed_path) as img:
                text = pytesseract.image_to_string(img)

        elif OCR_ENGINE == "gemini":
            import PIL.Image
            with PIL.Image.open(processed_path) as img:
                response = gemini_model.generate_content(
                    ["Extract all readable text from this document image:", img]
                )
                text = response.text.strip()

        else:
            raise ValueError(f"Unsupported OCR_ENGINE: {OCR_ENGINE}")

        # Safe file deletion with retry
        if os.path.exists(processed_path):
            try:
                import time
                time.sleep(0.1)  # Brief delay
                os.remove(processed_path)
            except PermissionError:
                print(f"Warning: Could not delete temporary file {processed_path}")

        return text

    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def safe_temp_cleanup():
    """Clean up temporary files safely"""
    upload_folder = app.config['UPLOAD_FOLDER']
    import time
    time.sleep(0.1)  # Brief delay to ensure file handles are released
    
    for fname in os.listdir(upload_folder):
        if '_processed.' in fname:  # Only clean up processed temp files
            fpath = os.path.join(upload_folder, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except (PermissionError, FileNotFoundError) as e:
                    print(f"Warning: Could not delete temp file {fpath}: {e}")



def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""
    

def classify_document(text):
    prompt = f"""
    Analyze the following document text and classify it into ONE of these exact categories:
    {", ".join(CATEGORIES)}

    If none of these categories match, respond with exactly: unknown

    Document text:
    {text[:2000]}  # Limit text length to avoid token limits

    Respond with only the category name, nothing else.
    """
    raw_output = query_model(prompt).strip().lower()
    
    # Clean the output to extract just the category
    for category in CATEGORIES:
        if category.lower() in raw_output.lower():
            return category
    
    return "unknown"


def extract_details(text, doc_type):
    prompt = f"""
    Please summarize the key fields and values found in this document text. Return the result as a valid JSON object.

    Document text:
    {text[:2000]}

    Respond with only a valid JSON object with field names and values, like this example:
    {{"Name": "John Doe", "Date of Birth": "01/01/1990"}}

    JSON:"""
    
    raw_output = query_model(prompt).strip()
    print(f'raw_ouput: {raw_output}')
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            details = json.loads(json_str)
            return details
        else:
            print(f"No JSON found in response: {raw_output}")
            return {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw output: {raw_output}")
        return {}


# Add PII categories after the CATEGORIES definition
PII_CATEGORIES = {
    "MNTI": {
        "SSN": "Social Security Number",
        "TIN": "Tax Identification Number", 
        "ITIN": "Individual Tax Identification Number",
        "EIN": "Employer Identification Number",
        "PASSPORT": "Passport Number",
        "DRIVER_LICENSE": "Driver's License Number",
        "NATIONAL_ID": "National ID Number"
    },
    "HCPI": {
        "MEDICAL_RECORD": "Medical Record Number",
        "INSURANCE_ID": "Health Insurance ID",
        "PATIENT_ID": "Patient ID",
        "PRESCRIPTION": "Prescription Number",
        "DIAGNOSIS": "Medical Diagnosis",
        "TREATMENT": "Treatment Information"
    },
    "General Personal Info": {
        "PERSON_NAME": "Person Name",
        "EMAIL": "Email Address",
        "PHONE": "Phone Number",
        "ADDRESS": "Physical Address",
        "CREDIT_CARD": "Credit Card Number",
        "BANK_ACCOUNT": "Bank Account Number",
        "DATE_OF_BIRTH": "Date of Birth",
        "AGE": "Age"
    },
    "Professional Info": {
        "EMPLOYEE_ID": "Employee ID",
        "SALARY": "Salary Information",
        "JOB_TITLE": "Job Title",
        "COMPANY": "Company Name",
        "DEPARTMENT": "Department"
    }
}

def detect_file_type(file_path):
    """Detect file type based on extension and content"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type:
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('audio/'):
            return 'audio'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type == 'application/pdf':
            return 'pdf'
        elif mime_type.startswith('text/'):
            return 'text'
    
    # Fallback to extension
    ext = file_path.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif']:
        return 'image'
    elif ext in ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg']:
        return 'audio'
    elif ext in ['mp4', 'avi', 'mov', 'mkv', 'wmv']:
        return 'video'
    elif ext == 'pdf':
        return 'pdf'
    elif ext in ['txt', 'rtf', 'doc', 'docx']:
        return 'text'
    
    return 'unknown'




def extract_text_with_coordinates(image_path, ocr_engine):
    """Extract text with bounding box coordinates using specified OCR engine"""
    try:
        processed_path = preprocess_image(image_path)
        
        if ocr_engine == "easyocr":
            # ... (EasyOCR code is fine) ...
            reader = get_ocr_reader(ocr_engine)
            result = reader.readtext(processed_path, detail=1)
            text_data = []
            full_text = ""
            for (bbox, text, confidence) in result:
                if confidence > 0.3:
                    x1, y1 = int(min([point[0] for point in bbox])), int(min([point[1] for point in bbox]))
                    x2, y2 = int(max([point[0] for point in bbox])), int(max([point[1] for point in bbox]))
                    text_data.append({
                        'text': text,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
                    full_text += text + " "
            
        elif ocr_engine == "tesseract":
            # ... (Tesseract code is fine) ...
            import pytesseract
            from PIL import Image
            
            with Image.open(processed_path) as img:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            text_data = []
            full_text = ""
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 20:
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        text_data.append({
                            'text': text,
                            'bbox': [x, y, x + w, y + h],
                            'confidence': data['conf'][i]
                        })
                        full_text += text + " "

        elif ocr_engine == "gemini":
            import google.generativeai as genai
            from google.generativeai.types import GenerationConfig
            import PIL.Image
            from google.genai.types import (
                GenerateContentConfig,
                HarmBlockThreshold,
                HarmCategory,
                HttpOptions,
                Part,
                
            )
            genai.configure(api_key=GEMINI_API_KEY)
            # This is the key difference: use `GenerationConfig` for JSON output.
            # `response_schema` is part of a different, more experimental API.
            config = GenerationConfig(
                response_mime_type="application/json",
            )
            
            prompt_parts = [
                "Extract all readable text from this document image along with its bounding box coordinates. The coordinates should be in the format [y_min, x_min, y_max, x_max] and normalized to a scale of 0 to 1. For example, [0.1, 0.2, 0.3, 0.4]. Return the response as a single JSON array of objects, where each object has 'text', 'bbox_norm', and 'confidence' (always 1.0).",
                PIL.Image.open(processed_path)
            ]
            
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            response = model.generate_content(
                prompt_parts,
                generation_config=config,
            )

            # --- Parse the JSON response ---
            text_data = []
            full_text = ""

            try:
                # The response.text will be the JSON string
                if response.text:
                    parsed_json = json.loads(response.text)
                    if isinstance(parsed_json, list):
                        img = PIL.Image.open(processed_path)
                        img_width, img_height = img.size
                        
                        for item in parsed_json:
                            # Convert normalized coordinates back to pixel coordinates
                            y_min_norm, x_min_norm, y_max_norm, x_max_norm = item['bbox_norm']
                            x_min = int(x_min_norm * img_width)
                            y_min = int(y_min_norm * img_height)
                            x_max = int(x_max_norm * img_width)
                            y_max = int(y_max_norm * img_height)
                            
                            text_data.append({
                                'text': item['text'],
                                'bbox': [x_min, y_min, x_max, y_max],
                                'confidence': 1.0  # Gemini doesn't provide confidence scores, so we use a default.
                            })
                            full_text += item['text'] + " "
                    else:
                        print("Debug: Gemini response was not a JSON list.")
                        # Fallback to text-only extraction
                        full_text = query_model(["Extract all readable text from this document image:", PIL.Image.open(processed_path)]).strip()
                        text_data = [{'text': full_text, 'bbox': None, 'confidence': 1.0}]
                else:
                    print("Debug: No text in Gemini response.")
                    full_text = ""
                    text_data = []
            except json.JSONDecodeError as e:
                print(f"JSON decode error with Gemini response: {e}")
                print(f"Raw response: {response.text}")
                full_text = query_model(["Extract all readable text from this document image:", PIL.Image.open(processed_path)]).strip()
                text_data = [{'text': full_text, 'bbox': None, 'confidence': 1.0}]

        else:
            raise ValueError(f"Unsupported OCR engine: {ocr_engine}")

        # Safe file deletion with retry
        if os.path.exists(processed_path):
            try:
                import time
                time.sleep(0.1)
                os.remove(processed_path)
            except PermissionError:
                print(f"Warning: Could not delete temporary file {processed_path}")

        return full_text.strip(), text_data

    except Exception as e:
        print(f"OCR with coordinates error: {e}")
        import traceback
        traceback.print_exc()
        return "", []

def transcribe_audio(file_path):
    """Transcribe audio file using Whisper"""
    try:
        # Load Whisper model (you can make this configurable)
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(file_path)
        return result["text"].strip()
        
    except Exception as e:
        print(f"Audio transcription error: {e}")
        return ""

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
            video.close()
            audio.close()
            return temp_audio.name
    except Exception as e:
        print(f"Video audio extraction error: {e}")
        return None

def detect_pii_in_text(text, selected_entities, custom_entities):
    """Use LLM to detect PII entities in text"""
    # Combine all selected entities
    all_entities = []
    for category, entities in PII_CATEGORIES.items():
        for entity_key, entity_name in entities.items():
            if entity_key in selected_entities:
                all_entities.append(entity_name)
    
    # Add custom entities
    if custom_entities:
        custom_list = [e.strip() for e in custom_entities.split(',') if e.strip()]
        all_entities.extend(custom_list)
    
    if not all_entities:
        return []
    
    entities_str = ", ".join(all_entities)
    
    prompt = f"""
    Analyze the following text and identify any PII (Personally Identifiable Information) entities.
    Look for these specific types: {entities_str}
    
    For each PII entity found, provide:
    1. The exact text that contains PII
    2. The type of PII
    3. A suggested redaction label (e.g., <SSN>, <NAME>, <EMAIL>)
    
    Text to analyze:
    {text[:3000]}  # Limit text length
    
    Respond ONLY with a JSON array in this exact format:
    [
        {{"text": "exact PII text found", "type": "PII type", "redaction": "<REDACTION_LABEL>"}},
        {{"text": "another PII text", "type": "PII type", "redaction": "<REDACTION_LABEL>"}}
    ]
    
    If no PII is found, respond with an empty array: []
    
    JSON:"""
    
    try:
        response = query_model(prompt).strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            pii_entities = json.loads(json_match.group())
            return pii_entities if isinstance(pii_entities, list) else []
        else:
            print(f"No JSON array found in PII detection response: {response}")
            return []
            
    except json.JSONDecodeError as e:
        print(f"PII detection JSON decode error: {e}")
        print(f"Response was: {response}")
        return []
    except Exception as e:
        print(f"PII detection error: {e}")
        return []

def create_redacted_image(image_path, text_data, pii_entities):
    """Create redacted image by drawing black boxes over PII text"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_copy = img.copy()
        
        # For each detected PII entity
        for pii in pii_entities:
            pii_text = pii['text'].lower().strip()
            
            # Find matching text blocks with coordinates
            for text_block in text_data:
                if text_block['bbox'] and pii_text in text_block['text'].lower():
                    x1, y1, x2, y2 = text_block['bbox']
                    # Draw black rectangle over the text
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        # Save redacted image
        redacted_path = image_path.replace('.', '_redacted.')
        cv2.imwrite(redacted_path, img_copy)
        return redacted_path
        
    except Exception as e:
        print(f"Image redaction error: {e}")
        return None

def create_redacted_text(original_text, pii_entities):
    """Create redacted text by replacing PII with redaction labels"""
    redacted_text = original_text
    
    # Sort PII entities by length (longest first) to avoid partial replacements
    pii_entities_sorted = sorted(pii_entities, key=lambda x: len(x['text']), reverse=True)
    
    for pii in pii_entities_sorted:
        original_pii_text = pii['text']
        redaction_label = pii['redaction']
        
        # Replace all occurrences (case insensitive)
        redacted_text = re.sub(re.escape(original_pii_text), redaction_label, redacted_text, flags=re.IGNORECASE)
    
    return redacted_text

# def create_redacted_image_with_coordinates(image_path, text_data, pii_entities):
#     """Create redacted image with precise coordinate mapping and save temporarily"""
#     try:
#         img = cv2.imread(image_path)
#         if img is None:
#             return None, []
        
#         img_copy = img.copy()
#         redaction_areas = []
        
#         print(f"Debug: Processing {len(pii_entities)} PII entities")
#         print(f"Debug: Available text blocks: {len(text_data)}")
        
#         # For each detected PII entity
#         for i, pii in enumerate(pii_entities):
#             pii_text = pii['text'].strip()
#             print(f"Debug: Looking for PII #{i+1}: '{pii_text}'")
            
#             found_match = False
            
#             # Find matching text blocks with coordinates
#             for j, text_block in enumerate(text_data):
#                 if text_block['bbox'] is None:
#                     continue
                    
#                 block_text = text_block['text'].strip()
#                 print(f"Debug: Checking against block #{j}: '{block_text}'")
                
#                 # Try different matching strategies
#                 # Strategy 1: Exact match (case insensitive)
#                 if pii_text.lower() in block_text.lower():
#                     x1, y1, x2, y2 = text_block['bbox']
#                     print(f"Debug: Found match in block #{j}, coordinates: ({x1}, {y1}, {x2}, {y2})")
                    
#                     # For now, redact the entire text block
#                     # You can add more precise sub-block redaction later
#                     cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    
#                     # Store redaction area for hover functionality
#                     img_height, img_width = img.shape[:2]
#                     redaction_areas.append({
#                         'x_percent': (x1 / img_width) * 100,
#                         'y_percent': (y1 / img_height) * 100,
#                         'width_percent': ((x2 - x1) / img_width) * 100,
#                         'height_percent': ((y2 - y1) / img_height) * 100,
#                         'pii_type': pii['type'],
#                         'pii_text': pii['text'],
#                         'redaction': pii['redaction']
#                     })
                    
#                     found_match = True
#                     print(f"Debug: Successfully redacted PII #{i+1}")
#                     break
                
#                 # Strategy 2: Fuzzy matching for partial words
#                 # Clean and normalize both texts for comparison
#                 clean_pii = ''.join(char.lower() for char in pii_text if char.isalnum())
#                 clean_block = ''.join(char.lower() for char in block_text if char.isalnum())
                
#                 if clean_pii and clean_pii in clean_block:
#                     x1, y1, x2, y2 = text_block['bbox']
#                     print(f"Debug: Found fuzzy match in block #{j}, coordinates: ({x1}, {y1}, {x2}, {y2})")
                    
#                     cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    
#                     img_height, img_width = img.shape[:2]
#                     redaction_areas.append({
#                         'x_percent': (x1 / img_width) * 100,
#                         'y_percent': (y1 / img_height) * 100,
#                         'width_percent': ((x2 - x1) / img_width) * 100,
#                         'height_percent': ((y2 - y1) / img_height) * 100,
#                         'pii_type': pii['type'],
#                         'pii_text': pii['text'],
#                         'redaction': pii['redaction']
#                     })
                    
#                     found_match = True
#                     print(f"Debug: Successfully redacted PII #{i+1} using fuzzy match")
#                     break
            
#             if not found_match:
#                 print(f"Debug: No match found for PII #{i+1}: '{pii_text}'")
#                 # Let's also print available text for debugging
#                 print("Debug: Available text blocks:")
#                 for k, text_block in enumerate(text_data[:5]):  # Show first 5 blocks
#                     print(f"  Block #{k}: '{text_block['text'][:50]}...'")
        
#         # Save redacted image temporarily with unique name
#         import uuid
#         temp_filename = f"redacted_{uuid.uuid4().hex[:8]}.png"
#         redacted_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
#         cv2.imwrite(redacted_path, img_copy)
        
#         # Store in session for cleanup
#         if 'temp_files' not in session:
#             session['temp_files'] = []
#         session['temp_files'].append(redacted_path)
        
#         print(f"Debug: Created redacted image with {len(redaction_areas)} redacted areas")
#         return redacted_path, redaction_areas
        
#     except Exception as e:
#         print(f"Image redaction error: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, []

def create_redacted_image_with_coordinates(image_path, text_data, pii_entities):
    """Create redacted image with precise coordinate mapping and save temporarily"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, []
        
        img_copy = img.copy()
        redaction_areas = []
        
        print(f"Debug: Processing {len(pii_entities)} PII entities")
        print(f"Debug: Available text blocks: {len(text_data)}")
        
        # For each detected PII entity
        for i, pii in enumerate(pii_entities):
            pii_text = pii['text'].strip()
            print(f"Debug: Looking for PII #{i+1}: '{pii_text}'")
            
            found_match = False
            
            # Find matching text blocks with coordinates
            for j, text_block in enumerate(text_data):
                if text_block['bbox'] is None:
                    print(f"Debug: Block #{j} has no coordinates (OCR engine doesn't support coordinates)")
                    continue
                    
                block_text = text_block['text'].strip()
                print(f"Debug: Checking against block #{j}: '{block_text}'")
                
                # Try different matching strategies
                # Strategy 1: Exact match (case insensitive)
                if pii_text.lower() in block_text.lower():
                    x1, y1, x2, y2 = text_block['bbox']
                    print(f"Debug: Found match in block #{j}, coordinates: ({x1}, {y1}, {x2}, {y2})")
                    
                    # For now, redact the entire text block
                    # You can add more precise sub-block redaction later
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    
                    # Store redaction area for hover functionality
                    img_height, img_width = img.shape[:2]
                    redaction_areas.append({
                        'x_percent': (x1 / img_width) * 100,
                        'y_percent': (y1 / img_height) * 100,
                        'width_percent': ((x2 - x1) / img_width) * 100,
                        'height_percent': ((y2 - y1) / img_height) * 100,
                        'pii_type': pii['type'],
                        'pii_text': pii['text'],
                        'redaction': pii['redaction']
                    })
                    
                    found_match = True
                    print(f"Debug: Successfully redacted PII #{i+1}")
                    break
                
                # Strategy 2: Fuzzy matching for partial words
                # Clean and normalize both texts for comparison
                clean_pii = ''.join(char.lower() for char in pii_text if char.isalnum())
                clean_block = ''.join(char.lower() for char in block_text if char.isalnum())
                
                if clean_pii and clean_pii in clean_block:
                    x1, y1, x2, y2 = text_block['bbox']
                    print(f"Debug: Found fuzzy match in block #{j}, coordinates: ({x1}, {y1}, {x2}, {y2})")
                    
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    
                    img_height, img_width = img.shape[:2]
                    redaction_areas.append({
                        'x_percent': (x1 / img_width) * 100,
                        'y_percent': (y1 / img_height) * 100,
                        'width_percent': ((x2 - x1) / img_width) * 100,
                        'height_percent': ((y2 - y1) / img_height) * 100,
                        'pii_type': pii['type'],
                        'pii_text': pii['text'],
                        'redaction': pii['redaction']
                    })
                    
                    found_match = True
                    print(f"Debug: Successfully redacted PII #{i+1} using fuzzy match")
                    break
            
            if not found_match:
                print(f"Debug: No match found for PII #{i+1}: '{pii_text}'")
                # Check if we have any blocks with coordinates
                blocks_with_coords = [t for t in text_data if t['bbox'] is not None]
                if not blocks_with_coords:
                    print("Debug: No text blocks have coordinates - OCR engine may not support coordinate extraction")
                else:
                    # Let's also print available text for debugging
                    print("Debug: Available text blocks:")
                    for k, text_block in enumerate(text_data[:5]):  # Show first 5 blocks
                        coord_info = f" at {text_block['bbox']}" if text_block['bbox'] else " (no coords)"
                        print(f"  Block #{k}: '{text_block['text'][:50]}...'{coord_info}")
        
        # Save redacted image temporarily with unique name
        import uuid
        temp_filename = f"redacted_{uuid.uuid4().hex[:8]}.png"
        redacted_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        cv2.imwrite(redacted_path, img_copy)
        
        # Store in session for cleanup
        if 'temp_files' not in session:
            session['temp_files'] = []
        session['temp_files'].append(redacted_path)
        
        print(f"Debug: Created redacted image with {len(redaction_areas)} redacted areas")
        return redacted_path, redaction_areas
        
    except Exception as e:
        print(f"Image redaction error: {e}")
        import traceback
        traceback.print_exc()
        return None, []
# =====================================
# FLASK APP
# =====================================

app = Flask(__name__)
app.secret_key = 'abc'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/process_document', methods=['POST'])
def process_document():
    try:
        if 'current_file' not in session:
            return jsonify({'error': 'No file to process'}), 400
            
        file_info = session['current_file']
        file_path = file_info['path']
        
        # Check if file still exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 400
        
        print(f"Processing file: {file_path}")  # Debug log
        
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        else:
            extracted_text = extract_text_from_image(file_path)
        
        print(f"Extracted text length: {len(extracted_text)}")  # Debug log
        print(f"Extracted text: {extracted_text}")  # Debug log
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            return jsonify({'error': 'Could not extract sufficient text from document'}), 400
        
        # Classify document
        doc_type = classify_document(extracted_text)
        print(f"Classified as: {doc_type}")  # Debug log
        
        # Extract details
        extracted_details = extract_details(extracted_text, doc_type)
        print(f"Extracted details: {extracted_details}")  # Debug log
        
        # Store results in session
        session['processing_results'] = {
            'document_type': doc_type,
            'extracted_text': extracted_text,
            'extracted_details': extracted_details,
            'processed_at': datetime.now().isoformat()
        }
        
        doc_type_display = doc_type.replace('_', ' ').title()
        return jsonify({
            'success': True,
            'document_type': doc_type_display,
            'confidence': 'High' if doc_type != 'unknown' else 'Low'
        })
        
    except Exception as e:
        print(f"Process document error: {str(e)}")  # Debug log
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Clean up all files in uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
        for fname in os.listdir(upload_folder):
            fpath = os.path.join(upload_folder, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception as cleanup_err:
                    print(f"Error deleting {fpath}: {cleanup_err}")


@app.route('/extract', methods=['POST'])
def extract_contents():
    try:
        if 'processing_results' not in session:
            return jsonify({'error': 'No processed document found'}), 400
        
        results = session['processing_results']
        
        return jsonify({
            'success': True,
            'details': results['extracted_details'],
            'document_type': results['document_type'].replace('_', ' ').title()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    try:
        # Clean up uploaded file
        if 'current_file' in session:
            file_path = session['current_file']['path']
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clean up temporary files
        if 'temp_files' in session:
            for temp_file in session['temp_files']:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        
        # Clear session
        session.clear()
        
        return jsonify({'success': True, 'message': 'Session cleared'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')

# PII Detection and Masking page route
@app.route('/pii')
def pii_detection():
    return render_template('pii.html')

# Document Classification and Extraction page route
@app.route('/classification')
def document_classification():
    return render_template('classification.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Store file info in session
            session['current_file'] = {
                'filename': filename,
                'original_name': file.filename,
                'path': file_path
            }
            
            return jsonify({'success': True, 'message': 'File uploaded successfully'})
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    try:
        global OCR_ENGINE, MODEL_PROVIDER, GEMINI_MODEL, GEMINI_API_KEY
        
        data = request.get_json()
        old_api_key = GEMINI_API_KEY
        
        

        # Update global variables
        OCR_ENGINE = data.get('ocr_engine', OCR_ENGINE)
        MODEL_PROVIDER = data.get('llm_provider', MODEL_PROVIDER)
        GEMINI_API_KEY = data.get('gemini_api_key', GEMINI_API_KEY)
        
        if GEMINI_API_KEY != old_api_key:
            print(f"GEMINI_API_KEY updated.")
        # Update Gemini model based on which service is using it
        if OCR_ENGINE == 'gemini':
            GEMINI_MODEL = data.get('ocr_gemini_variant', GEMINI_MODEL)
        elif MODEL_PROVIDER == 'api':
            GEMINI_MODEL = data.get('llm_gemini_variant', GEMINI_MODEL)
        
        print(f"Settings updated - OCR: {OCR_ENGINE}, LLM: {MODEL_PROVIDER}, Gemini: {GEMINI_MODEL}")
        
        return jsonify({'success': True, 'message': 'Settings updated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/process_pii', methods=['POST'])
# def process_pii():
#     try:
#         if 'current_file' not in session:
#             return jsonify({'error': 'No file to process'}), 400
            
#         file_info = session['current_file']
#         file_path = file_info['path']
        
#         # Get selected PII entities from request
#         data = request.get_json()
#         selected_entities = data.get('selected_entities', [])
#         custom_entities = data.get('custom_entities', '')
        
#         if not selected_entities and not custom_entities:
#             return jsonify({'error': 'Please select at least one PII entity type to detect'}), 400
        
#         # Check if file still exists
#         if not os.path.exists(file_path):
#             return jsonify({'error': 'File not found'}), 400
        
#         # Detect file type
#         file_type = detect_file_type(file_path)
#         print(f"Processing {file_type} file: {file_path}")
        
#         extracted_text = ""
#         text_data = []
        
#         # Extract text based on file type
#         if file_type == 'image':
#             extracted_text, text_data = extract_text_with_coordinates(file_path, OCR_ENGINE)
#         elif file_type == 'pdf':
#             extracted_text = extract_text_from_pdf(file_path)
#         elif file_type == 'text':
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 extracted_text = f.read()
#         elif file_type == 'audio':
#             extracted_text = transcribe_audio(file_path)
#         elif file_type == 'video':
#             # Extract audio from video first
#             audio_path = extract_audio_from_video(file_path)
#             if audio_path:
#                 extracted_text = transcribe_audio(audio_path)
#                 try:
#                     os.remove(audio_path)  # Clean up temp audio file
#                 except:
#                     pass
#         else:
#             return jsonify({'error': f'Unsupported file type: {file_type}'}), 400
        
#         if not extracted_text or len(extracted_text.strip()) < 10:
#             return jsonify({'error': 'Could not extract sufficient text from file'}), 400
        
#         # Detect PII in extracted text
#         pii_entities = detect_pii_in_text(extracted_text, selected_entities, custom_entities)
        
#         # Store results in session
#         session['pii_results'] = {
#             'file_type': file_type,
#             'extracted_text': extracted_text,
#             'text_data': text_data,
#             'pii_entities': pii_entities,
#             'selected_entities': selected_entities,
#             'custom_entities': custom_entities,
#             'processed_at': datetime.now().isoformat()
#         }
        
#         pii_count = len(pii_entities)
#         return jsonify({
#             'success': True,
#             'file_type': file_type.title(),
#             'pii_count': pii_count,
#             'pii_found': pii_count > 0
#         })
        
#     except Exception as e:
#         print(f"Process PII error: {str(e)}")
#         return jsonify({'error': f'Processing failed: {str(e)}'}), 500
#     finally:
#         # Clean up temporary processed files
#         safe_temp_cleanup()

@app.route('/process_pii', methods=['POST'])
def process_pii():
    try:
        if 'current_file' not in session:
            return jsonify({'error': 'No file to process'}), 400

        file_info = session['current_file']
        file_path = file_info['path']

        # Get selected PII entities from request
        data = request.get_json()
        selected_entities = data.get('selected_entities', [])
        custom_entities = data.get('custom_entities', '')

        if not selected_entities and not custom_entities:
            return jsonify({'error': 'Please select at least one PII entity type to detect'}), 400

        # Check if file still exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 400

        # Detect file type
        file_type = detect_file_type(file_path)
        print(f"Processing {file_type} file: {file_path}")

        # Prepare labels for PII detection
        labels = selected_entities + [e.strip() for e in custom_entities.split(',') if e.strip()]

        if file_type == 'image':
            # Use EasyOCR from common_functions
            import cv2
            img = cv2.imread(file_path)
            if img is None:
                return jsonify({'error': 'Could not read image file'}), 400
                
            detections = reader.readtext(file_path)
            # detections: list of (bbox, text, confidence)
            
            # Convert detections to expected format for draw_black_rectangles
            processed_detections = [(d[0], d[1], d[2]) for d in detections]
            
            # Get full text for PII identification
            full_text = " ".join([d[1] for d in processed_detections])
            
            # Identify PII
            pii_entities = identify_pii(full_text, labels, gliner_model, llm_model, spacy_model)
            
            # Redact image
            draw_black_rectangles(img, processed_detections, labels, gliner_model, llm_model, spacy_model)
            
            # Save redacted image
            import uuid
            redacted_filename = f"redacted_{uuid.uuid4().hex[:8]}.png"
            redacted_path = os.path.join(app.config['UPLOAD_FOLDER'], redacted_filename)
            cv2.imwrite(redacted_path, img)
            
            # Store in session for cleanup
            if 'temp_files' not in session:
                session['temp_files'] = []
            session['temp_files'].append(redacted_path)
            
            session['pii_results'] = {
                'file_type': file_type,
                'redacted_image_path': redacted_path,
                'redacted_image_filename': redacted_filename,
                'extracted_text': full_text,
                'pii_entities': [{'text': ent[0], 'type': ent[1], 'redaction': f'<{ent[1]}>'} for ent in pii_entities],
                'selected_entities': selected_entities,
                'custom_entities': custom_entities,
                'processed_at': datetime.now().isoformat()
            }
            
            pii_count = len(pii_entities)
            return jsonify({
                'success': True,
                'file_type': file_type.title(),
                'pii_count': pii_count,
                'pii_found': pii_count > 0
            })

        elif file_type == 'text':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
                
            pii_entities = identify_pii(extracted_text, labels, gliner_model, llm_model, spacy_model)
            redacted_text = redact_pii(extracted_text, labels, gliner_model, llm_model, spacy_model)
            
            session['pii_results'] = {
                'file_type': file_type,
                'extracted_text': extracted_text,
                'redacted_text': redacted_text,
                'pii_entities': [{'text': ent[0], 'type': ent[1], 'redaction': f'<{ent[1]}>'} for ent in pii_entities],
                'selected_entities': selected_entities,
                'custom_entities': custom_entities,
                'processed_at': datetime.now().isoformat()
            }
            
            pii_count = len(pii_entities)
            return jsonify({
                'success': True,
                'file_type': file_type.title(),
                'pii_count': pii_count,
                'pii_found': pii_count > 0
            })

        elif file_type == 'audio':
            # Use your existing audio transcription logic, then pass text to identify_pii/redact_pii
            extracted_text = transcribe_audio(file_path)
            if not extracted_text:
                return jsonify({'error': 'Could not transcribe audio file'}), 400
                
            pii_entities = identify_pii(extracted_text, labels, gliner_model, llm_model, spacy_model)
            redacted_text = redact_pii(extracted_text, labels, gliner_model, llm_model, spacy_model)
            
            session['pii_results'] = {
                'file_type': file_type,
                'extracted_text': extracted_text,
                'redacted_text': redacted_text,
                'pii_entities': [{'text': ent[0], 'type': ent[1], 'redaction': f'<{ent[1]}>'} for ent in pii_entities],
                'selected_entities': selected_entities,
                'custom_entities': custom_entities,
                'processed_at': datetime.now().isoformat()
            }
            
            pii_count = len(pii_entities)
            return jsonify({
                'success': True,
                'file_type': file_type.title(),
                'pii_count': pii_count,
                'pii_found': pii_count > 0
            })

        elif file_type == 'pdf':
            extracted_text = extract_text_from_pdf(file_path)
            if not extracted_text:
                return jsonify({'error': 'Could not extract text from PDF'}), 400
                
            pii_entities = identify_pii(extracted_text, labels, gliner_model, llm_model, spacy_model)
            redacted_text = redact_pii(extracted_text, labels, gliner_model, llm_model, spacy_model)
            
            session['pii_results'] = {
                'file_type': file_type,
                'extracted_text': extracted_text,
                'redacted_text': redacted_text,
                'pii_entities': [{'text': ent[0], 'type': ent[1], 'redaction': f'<{ent[1]}>'} for ent in pii_entities],
                'selected_entities': selected_entities,
                'custom_entities': custom_entities,
                'processed_at': datetime.now().isoformat()
            }
            
            pii_count = len(pii_entities)
            return jsonify({
                'success': True,
                'file_type': file_type.title(),
                'pii_count': pii_count,
                'pii_found': pii_count > 0
            })

        else:
            return jsonify({'error': f'Unsupported file type: {file_type}'}), 400

    except Exception as e:
        print(f"Process PII error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        safe_temp_cleanup()

@app.route('/get_redacted_image/<filename>')
def get_redacted_image(filename):
    """Serve the temporarily saved redacted image"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            from flask import send_file
            return send_file(file_path, mimetype='image/png')
        else:
            return "Image not found", 404
    except Exception as e:
        return str(e), 500

# @app.route('/get_redacted_content', methods=['POST'])
# def get_redacted_content():
#     try:
#         if 'pii_results' not in session:
#             return jsonify({'error': 'No PII processing results found'}), 400
            
#         if 'current_file' not in session:
#             return jsonify({'error': 'No file found'}), 400
        
#         results = session['pii_results']
#         file_info = session['current_file']
#         file_path = file_info['path']
        
#         file_type = results['file_type']
#         extracted_text = results['extracted_text']
#         text_data = results['text_data']
#         pii_entities = results['pii_entities']
        
#         response_data = {
#             'success': True,
#             'file_type': file_type,
#             'pii_entities': pii_entities,
#             'original_text': extracted_text
#         }
        
#         if file_type == 'image':
#             # Create redacted image and save temporarily
#             redacted_image_path, redaction_areas = create_redacted_image_with_coordinates(file_path, text_data, pii_entities)
            
#             if redacted_image_path and os.path.exists(redacted_image_path):
#                 # Convert original image to base64
#                 with open(file_path, 'rb') as f:
#                     original_image_b64 = base64.b64encode(f.read()).decode()
                
#                 # Use URL for redacted image instead of base64
#                 redacted_filename = os.path.basename(redacted_image_path)
                
#                 response_data.update({
#                     'original_image': f"data:image/jpeg;base64,{original_image_b64}",
#                     'redacted_image_url': f"/get_redacted_image/{redacted_filename}",
#                     'redaction_areas': redaction_areas
#                 })
                
#             else:
#                 return jsonify({'error': 'Failed to create redacted image'}), 500
                
#         else:
#             # For text-based content, create redacted text
#             redacted_text = create_redacted_text(extracted_text, pii_entities)
#             response_data['redacted_text'] = redacted_text
            
#             # For audio/video, also provide file path for audio player
#             if file_type in ['audio', 'video']:
#                 try:
#                     with open(file_path, 'rb') as f:
#                         audio_b64 = base64.b64encode(f.read()).decode()
#                     response_data['audio_data'] = f"data:audio/mpeg;base64,{audio_b64}"
#                 except:
#                     response_data['audio_data'] = None
        
#         return jsonify(response_data)
        
#     except Exception as e:
#         print(f"Get redacted content error: {str(e)}")
#         return jsonify({'error': f'Failed to create redacted content: {str(e)}'}), 500

@app.route('/get_redacted_content', methods=['POST'])
def get_redacted_content():
    try:
        if 'pii_results' not in session:
            return jsonify({'error': 'No PII processing results found'}), 400
            
        if 'current_file' not in session:
            return jsonify({'error': 'No file found'}), 400
        
        results = session['pii_results']
        file_info = session['current_file']
        file_path = file_info['path']
        
        file_type = results['file_type']
        extracted_text = results.get('extracted_text', '')
        pii_entities = results['pii_entities']
        
        response_data = {
            'success': True,
            'file_type': file_type,
            'pii_entities': pii_entities,
            'original_text': extracted_text
        }
        
        if file_type == 'image':
            # Use the saved redacted image
            redacted_filename = results.get('redacted_image_filename')
            if redacted_filename:
                # Convert original image to base64
                with open(file_path, 'rb') as f:
                    original_image_b64 = base64.b64encode(f.read()).decode()
                
                response_data.update({
                    'original_image': f"data:image/jpeg;base64,{original_image_b64}",
                    'redacted_image_url': f"/get_redacted_image/{redacted_filename}",
                    'redaction_areas': []  # You can implement hover areas later if needed
                })
            else:
                return jsonify({'error': 'Redacted image not found'}), 500
                
        else:
            # For text-based content, use the redacted text from results
            redacted_text = results.get('redacted_text', extracted_text)
            response_data['redacted_text'] = redacted_text
            
            # For audio/video, also provide file path for audio player
            if file_type in ['audio', 'video']:
                try:
                    with open(file_path, 'rb') as f:
                        audio_b64 = base64.b64encode(f.read()).decode()
                    response_data['audio_data'] = f"data:audio/mpeg;base64,{audio_b64}"
                except:
                    response_data['audio_data'] = None
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Get redacted content error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to create redacted content: {str(e)}'}), 500


@app.route('/get_pii_categories', methods=['GET'])
def get_pii_categories():
    """Return PII categories for frontend"""
    return jsonify({
        'success': True,
        'categories': PII_CATEGORIES
    })

ALLOWED_EXTENSIONS.update({'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'mp4', 'avi', 'mov', 'mkv', 'wmv', 'txt', 'rtf', 'doc', 'docx'})

if __name__ == "__main__":
    app.run(debug=True)
