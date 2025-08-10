# =====================================
# CONFIGURATION BLOCK
# =====================================

# --- LLM Provider Settings ---
MODEL_PROVIDER = "api"  # options: "huggingface", "ollama", "api"
MODEL_NAME = "gemma3:4b"  # e.g., "gemma-7b-it", "llama3:8b", "mistral-7b-instruct"
HF_LOCAL_PATH = "/path/to/huggingface/model"  # used if MODEL_PROVIDER = "huggingface"

# --- OCR Settings ---
OCR_ENGINE = "gemini"  # options: "easyocr", "paddleocr", "tesseract", "gemini"
OCR_LANGUAGES = ['en']

# --- Gemini API Settings ---
GEMINI_API_KEY = "AIzaSyC1UyxGaDx7j2caQz_F5XYy6-08rMYzJ8Q"  # or use env var
GEMINI_MODEL = "gemini-2.0-flash-lite"  # or "gemini-1.5-flash-vision"

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
# LLM Init
if MODEL_PROVIDER == "huggingface":
    from transformers import pipeline
    if os.path.exists(HF_LOCAL_PATH):
        llm_pipe = pipeline("text-generation", model=HF_LOCAL_PATH, device_map="auto")
    else:
        llm_pipe = pipeline("text-generation", model=MODEL_NAME, device_map="auto")

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
            result = reader.ocr(processed_path, cls=True)
            text = "\n".join([line[1][0] for page in result for line in page]).strip()

        elif ocr_engine == "tesseract":
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(processed_path))

        elif ocr_engine == "gemini":
            import PIL.Image
            model = get_ocr_reader(ocr_engine)
            img = PIL.Image.open(processed_path)
            response = model.generate_content(
                ["Extract all readable text from this document image:", img]
            )
            text = response.text.strip()

        else:
            raise ValueError(f"Unsupported OCR engine: {ocr_engine}")

        if os.path.exists(processed_path):
            os.remove(processed_path)

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
        ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

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
            result = ocr_reader.ocr(processed_path, cls=True)
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

        if os.path.exists(processed_path):
            os.remove(processed_path)

        return text

    except Exception as e:
        print(f"OCR error: {e}")
        return ""

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
        
        # Clear session
        session.clear()
        
        return jsonify({'success': True, 'message': 'Session cleared'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')

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
        global OCR_ENGINE, MODEL_PROVIDER, GEMINI_MODEL
        
        data = request.get_json()
        
        # Update global variables
        OCR_ENGINE = data.get('ocr_engine', OCR_ENGINE)
        MODEL_PROVIDER = data.get('llm_provider', MODEL_PROVIDER)
        
        # Update Gemini model based on which service is using it
        if OCR_ENGINE == 'gemini':
            GEMINI_MODEL = data.get('ocr_gemini_variant', GEMINI_MODEL)
        elif MODEL_PROVIDER == 'api':
            GEMINI_MODEL = data.get('llm_gemini_variant', GEMINI_MODEL)
        
        print(f"Settings updated - OCR: {OCR_ENGINE}, LLM: {MODEL_PROVIDER}, Gemini: {GEMINI_MODEL}")
        
        return jsonify({'success': True, 'message': 'Settings updated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
