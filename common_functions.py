import json
import re
import cv2
import spacy
import os
from gliner import GLiNER
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file
from werkzeug.utils import secure_filename
import easyocr
import subprocess
import importlib.util

def get_gliner_model(local_path="models/gliner_medium-v2.1", model_name="urchade/gliner_medium-v2.1"):
    # Check if config.json exists in local path
    config_path = os.path.join(local_path, "config.json")

    if os.path.exists(config_path):
        print(f"Loading GLiNER model from local path: {local_path}")
        return GLiNER.from_pretrained(local_path)

    # Otherwise download fresh copy from Hugging Face
    print(f"Downloading GLiNER model from Hugging Face: {model_name}")
    model = GLiNER.from_pretrained(model_name)
    os.makedirs(local_path, exist_ok=True)
    model.save_pretrained(local_path)
    return model

def get_spacy_model(model_name="en_core_web_sm"):
    try:
        # Try loading directly
        print(f"Loading SpaCy model: {model_name}")
        return spacy.load(model_name)
    except OSError:
        # If not installed, download it
        print(f"SpaCy model '{model_name}' not found. Downloading...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

print("Initializing models...")
# gliner_model = get_gliner_model()
# spacy_model = get_spacy_model()
reader = easyocr.Reader(['en'], model_storage_directory="./models/easyocr/")
# llm_model = Ollama(model="llama3.2")
print("Models initialized successfully.")



def load_definitions(file_path="definitions.json"):
    try:
        with open(file_path, "r") as file:
            definitions = json.load(file)
        print(f"Loaded definitions from {file_path}")
        return definitions
    except Exception as e:
        print(f"Failed to load definitions from {file_path}: {e}")
        return {}

def detect_pii_with_llm(text, labels, definitions, llm_model):
    instructions = []
    for label in labels:
        if label in definitions:
            instructions.append(f"{label}: {definitions[label]['description']}")
        else:
            instructions.append(f"Find all instances of {label} in the text.")

    instruction_text = "You are a PII extraction model. Identify and label the following types of PII:\n"
    instruction_text += "\n".join(instructions)

    prompt = (
        f"{instruction_text}\n\n"
        f"Text:\n{text}\n\n"
        'Return only a valid JSON dictionary of the format:\n'
        '{"label1": ["value1", "value2"], "label2": ["value3"]}\n'
        "Do not include code, explanation, or any other text. Just the dictionary."
    )

    print(f"Sending prompt to LLM:\n{prompt[:1000]}...")

    try:
        response = llm_model.invoke(prompt)

        response_text = (
            response if isinstance(response, str)
            else response.content if hasattr(response, "content")
            else str(response)
        )

        print(f"LLM raw response: {response_text[:1000]}...")

        if "def " in response_text or "import " in response_text:
            raise ValueError("LLM returned code instead of JSON")

        match = re.search(r"{\s*.*?}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON dictionary found in LLM response")

        json_str = match.group(0)
        result_dict = json.loads(json_str)

    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        result_dict = {}

    result = [
        (str(entity_text), label)
        for label, entities in result_dict.items()
        if isinstance(entities, list)
        for entity_text in entities
        if isinstance(entity_text, (str, int, float))
    ]

    print(f"LLM detected {len(result)} PII entities")
    return result


def identify_pii(text, labels, gliner_model, llm_model, spacy_model):
    print("Running PII detection using SpaCy, GLiNER, and LLM...")
    definitions = load_definitions()

    pii_entities_spacy = [(ent.text, ent.label_) for ent in spacy_model(text).ents if ent.label_ in labels]
    print(f"SpaCy detected {len(pii_entities_spacy)} entities")

    gliner_entities = gliner_model.predict_entities(text, labels)
    pii_entities_gliner = [(entity["text"], entity["label"]) for entity in gliner_entities]
    print(f"GLiNER detected {len(pii_entities_gliner)} entities")

    pii_entities_llm = detect_pii_with_llm(text, labels, definitions, llm_model)

    pii_entities = set(pii_entities_spacy + pii_entities_gliner + pii_entities_llm)
    print(f"Total unique PII entities identified: {len(pii_entities)}")
    return pii_entities


def redact_pii(text, labels, gliner_model, llm_model, spacy_model):
    print("Redacting PII from text...")
    pii_entities = identify_pii(text, labels, gliner_model, llm_model, spacy_model)
    redacted_text = text

    for ent_text, ent_label in pii_entities:
        redacted_text = redacted_text.replace(ent_text, f"<{ent_label}>")

    print("Text redaction completed")
    return redacted_text


def draw_black_rectangles(image, detections, labels, gliner_model, llm_model, spacy_model):
    print("Drawing black rectangles on image for detected PII...")
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # thickness = 1
    tolerance = 0.15

    full_text = ", ".join([d[1] for d in detections])
    pii_entities = identify_pii(full_text, labels, gliner_model, llm_model, spacy_model)

    for coordinates, text, _ in detections:
        top_left = (int(coordinates[0][0]), int(coordinates[0][1]))
        bottom_right = (int(coordinates[2][0]), int(coordinates[2][1]))

        for pii_text, _ in pii_entities:
            if pii_text in text:
                entity_start = text.find(pii_text)
                entity_end = entity_start + len(pii_text)

                entity_start_fraction = entity_start / len(text)
                entity_end_fraction = entity_end / len(text)

                entity_start_x = int(top_left[0] + (bottom_right[0] - top_left[0]) * entity_start_fraction)
                entity_end_x = int(top_left[0] + (bottom_right[0] - top_left[0]) * entity_end_fraction)

                entity_start_x = max(top_left[0], int(entity_start_x - (bottom_right[0] - top_left[0]) * tolerance))
                entity_end_x = min(bottom_right[0], int(entity_end_x + (bottom_right[0] - top_left[0]) * tolerance))

                cv2.rectangle(image, (entity_start_x, top_left[1]), (entity_end_x, bottom_right[1]), (0, 0, 0), thickness=-1)

    print("Black rectangles drawn for image redaction")
