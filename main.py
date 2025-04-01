from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import uuid
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

# Create necessary directories
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# YOLO Model Path
YOLO_MODEL_PATH = "/Users/shubhampund9767/FinalYrProject/FinalYrProject/best1.pt"  # Ensure this is the correct relative path

# Disease Information Dictionary
disease_info = {
    0: {"name": "Aphid", "description": "Small, soft-bodied insects that suck sap.", "precaution": "Use resistant varieties, introduce natural predators.", "medicine": "Neem oil, Imidacloprid."},
    1: {"name": "Aphids", "description": "Rapidly multiplying sap-sucking pests.", "precaution": "Field monitoring, remove infected plants.", "medicine": "Pyrethroid-based insecticides."},
    2: {"name": "Army Worm", "description": "Caterpillars that chew leaves.", "precaution": "Pheromone traps, biological control.", "medicine": "Bacillus thuringiensis, Spinosad."},
    3: {"name": "Bacterial Blight", "description": "Water-soaked lesions leading to wilting.", "precaution": "Use disease-free seeds, crop rotation.", "medicine": "Copper-based bactericides, Streptomycin."},
    4: {"name": "Cotton Boll Rot", "description": "Fungal infection causing decay.", "precaution": "Ensure aeration, remove infected bolls.", "medicine": "Carbendazim-based fungicides."},
    5: {"name": "Green Cotton Boll", "description": "Immature bolls remain green.", "precaution": "Proper irrigation, control bollworms.", "medicine": "Mancozeb fungicide, growth regulators."},
    6: {"name": "Healthy", "description": "No disease detected.", "precaution": "Maintain good soil fertility and proper crop rotation.", "medicine": "No treatment needed."},
    7: {"name": "Powdery Mildew", "description": "White powdery fungal spots on leaves.", "precaution": "Avoid humidity, remove infected leaves.", "medicine": "Sulfur-based fungicides, Tebuconazole."},
    8: {"name": "Target Spot", "description": "Circular brown spots leading to defoliation.", "precaution": "Improve drainage, avoid high density.", "medicine": "Chlorothalonil, copper-based fungicides."}
}

# Configure Gemini AI
genai.configure(api_key="AIzaSyDP2CIAmXMjUZm8okUksFR5_qcgFJj8vGM")  # Use environment variable for security

# Load YOLO Model
def initialize_model():
    """Initialize YOLO model with error handling."""
    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
        model = YOLO(YOLO_MODEL_PATH)
        model.conf = 0.25
        model.iou = 0.45
        print("YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        raise

# Initialize YOLO Model
yolo_model = initialize_model()

def generate_gemini_response(detections):
    """Generate AI-powered disease analysis using Gemini AI."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        if detections and detections[0]['class'] != "Healthy":
            diseases_text = "\n".join(
                f"- Disease: {det['class']}\n  Confidence: {det['confidence']:.2f}%\n  Symptoms: {det['description']}"
                for det in detections
            )
            
            prompt = f"""
                     You are a highly skilled agricultural expert specializing in cotton disease management. 
                    Analyze the following detected cotton diseases in detail:

                    {diseases_text}

                   Provide a comprehensive expert analysis covering the following aspects:

                    ### 1. Severity of Infection  
                    - Clearly categorize the infection level (Mild, Moderate, Severe).  
                     - Explain how quickly it can spread and affect the crop.  

                    ### 2. Immediate Actions for the Farmer  
                   - Step-by-step guidance on what the farmer should do immediately.  
                    - Highlight any quarantine measures, irrigation changes, or pruning techniques.  

                    ### 3. Economic Impact if Untreated  
                    - Project potential yield loss and financial damage.  
                    - Provide real-world examples or statistics if available.  

                    ### 4. Best Treatment Options  
                    - Recommend specific pesticides, fungicides, or organic treatments.  
                    - Mention dosage and application frequency.  

                    ### 5. Prevention Methods  
                    - Suggest crop rotation strategies, soil health improvement, and resistant seed varieties.  
                    - Highlight seasonal precautions to minimize future risks.  

                    ### Response Format:  
                    Return the analysis in structured HTML format for easy readability. Ensure:  
                    - **Bold headings** for sections  
                    - **Bullet points** for clarity  
                    - **Tables where needed** to present data concisely  

                    Your response should be insightful, actionable, and easy to interpret by farmers and agronomists.
                    """


            response = model.generate_content(prompt)
            return response.text if response else "Unable to generate AI analysis"
        
        return "<div class='healthy-message'>No diseases detected. The plant appears healthy.</div>"

    except Exception as e:
        return f"<div class='error-message'>Error generating AI analysis: {str(e)}</div>"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle disease detection and AI analysis."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    if not image.filename:
        return jsonify({'error': 'No selected file'}), 400
    
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    processed_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
    
    try:
        image.save(image_path)
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        results = yolo_model.predict(source=img, save=False, verbose=False)
        detections = []
        result = results[0]
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                
                if conf > 0.25 and class_id in disease_info:
                    class_data = disease_info[class_id]
                    detections.append({
                        "class": class_data["name"],
                        "confidence": round(conf * 100, 2),
                        "box": [x1, y1, x2, y2],
                        "description": class_data["description"],
                        "precaution": class_data["precaution"],
                        "medicine": class_data["medicine"]
                    })
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{class_data['name']} ({conf*100:.2f}%)", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 255, 0), 2)
        
        if not detections:
            detections.append(disease_info[6])  # Default to "Healthy" if no diseases detected

        cv2.imwrite(processed_path, img)

        gemini_response = generate_gemini_response(detections)

        return jsonify({
            'status': 'success',
            'detections': detections,
            'processed_image': f"processed_{filename}",
            'gemini_response': gemini_response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)