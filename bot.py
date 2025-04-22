import os
import subprocess
import time
import requests
from flask import Flask, render_template, request, jsonify, redirect
from flask_ngrok import run_with_ngrok
import google.generativeai as genai
from flask_sqlalchemy import SQLAlchemy
from PyPDF2 import PdfReader
import json
from diffusers import StableDiffusionPipeline
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory for Colab
BASE_DIR = "/content/AI-fbx"
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Create directories if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Initialize Flask app with explicit static and template folders
app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
run_with_ngrok(app)

# Setup ngrok
def setup_ngrok():
    try:
        subprocess.run(
            "curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null",
            shell=True, check=True
        )
        subprocess.run(
            'echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list',
            shell=True, check=True
        )
        subprocess.run("sudo apt update && sudo apt install ngrok -y", shell=True, check=True)
        authtoken = "2n1l3KxTC23zFLnuM94ryL284Wp_TJuu2gPEffpuYqvM2q59"
        subprocess.run(f"ngrok config add-authtoken {authtoken}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up ngrok: {e}")
        raise

def start_ngrok():
    try:
        ngrok_process = subprocess.Popen(["ngrok", "http", "5000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Wait for ngrok to initialize
        return ngrok_process
    except Exception as e:
        logger.error(f"Error starting ngrok: {e}")
        raise

# Configure Stable Diffusion Pipeline
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("cuda")  # Move to GPU
except Exception as e:
    logger.error(f"Error initializing Stable Diffusion pipeline: {e}")
    raise

# Configure Gemini API
genai.configure(api_key="AIzaSyCcSn22t65ApHqRohShr7lefdbgS9icU2M")

# Configure SQLite Database for Chat History
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'chat_history.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String, nullable=False)
    ai_response = db.Column(db.String, nullable=False)

with app.app_context():
    db.create_all()

# Summarizer Configuration
summarizer_generation_config = {
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}
summarizer_model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=summarizer_generation_config)

# Chatbot Function using Gemini
def get_ai_response(user_input):
    try:
        chat_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = chat_model.generate_content(user_input)
        return response.text
    except Exception as e:
        logger.error(f"Error in Gemini response: {e}")
        return "Error getting AI response"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = ""
        if 'message' in request.form and request.form['message']:
            user_message = request.form['message']
        elif 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename.endswith('.pdf'):
                try:
                    pdf_reader = PdfReader(uploaded_file)
                    user_message = " ".join(page.extract_text() for page in pdf_reader.pages)
                except Exception as e:
                    logger.error(f"Error reading PDF: {e}")
                    return jsonify(error="Error reading PDF file")
            else:
                return jsonify(error="Unsupported file format. Please upload a PDF.")
        else:
            return jsonify(error="No message or file provided")

        ai_response = get_ai_response(user_message)
        new_chat = ChatHistory(user_message=user_message, ai_response=ai_response)
        db.session.add(new_chat)
        db.session.commit()
        return jsonify(response=ai_response)
    return render_template('chatbot1.html')

@app.route('/summarizer', methods=['GET', 'POST'])
def summarizer():
    if request.method == 'POST':
        if 'text' in request.form:
            user_text = request.form['text']
        elif 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(uploaded_file)
                user_text = " ".join(page.extract_text() for page in pdf_reader.pages)
            else:
                return jsonify(error="Unsupported file format. Please upload a PDF.")
        try:
            summary_response = summarizer_model.generate_content(f"Summarize the following text: {user_text}")
            ai_summary = summary_response.text
            return jsonify(summary=ai_summary)
        except Exception as e:
            logger.error(f"Error in summarizer: {e}")
            return jsonify(error=str(e))
    return render_template('chatbot2.html')

# Updated Image Generator Route using Stable Diffusion
@app.route('/generate_image', methods=['POST'])
def generate_image():
    user_prompt = request.form['prompt']
    try:
        # Generate image using Stable Diffusion pipeline
        logger.info(f"Generating image for prompt: {user_prompt}")
        image = pipe(user_prompt).images[0]
        # Save the image to the static folder with a unique filename
        image_filename = f"generated_image_{int(time.time())}.png"
        image_path = os.path.join(STATIC_DIR, image_filename)
        image.save(image_path)
        logger.info(f"Image saved to: {image_path}")
        # Return the URL path for the frontend
        image_url = f"/static/{image_filename}"
        return jsonify(image_path=image_url)
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return jsonify(error=str(e))

@app.route('/imagebot3')
def imagebot_page():
    return render_template('imagebot3.html')

@app.route('/chat_history', methods=['GET'])
def chat_history():
    history = ChatHistory.query.all()
    return jsonify([{'user': chat.user_message, 'ai': chat.ai_response} for chat in history])

@app.route('/clear_chat', methods=['GET'])
def clear_chat():
    ChatHistory.query.delete()
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/delete_history', methods=['GET'])
def delete_history():
    ChatHistory.query.delete()
    db.session.commit()
    return jsonify({"status": "deleted"})

if __name__ == '__main__':
    setup_ngrok()  # Install and configure ngrok
    start_ngrok()  # Start ngrok tunnel
    logger.info("Starting Flask app")
    app.run()      # Run Flask app with ngrok
