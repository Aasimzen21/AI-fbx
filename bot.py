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
from diffusers import StableDiffusionPipeline  # Added for Stable Diffusion
import torch  # Added for torch support

app = Flask(__name__, template_folder='/content/AI-fbx/templates')
run_with_ngrok(app)

# Setup ngrok
def setup_ngrok():
    subprocess.run("curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null", shell=True, check=True)
    subprocess.run('echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list', shell=True, check=True)
    subprocess.run("sudo apt update && sudo apt install ngrok -y", shell=True, check=True)
    authtoken = "2n1l3KxTC23zFLnuM94ryL284Wp_TJuu2gPEffpuYqvM2q59"
    subprocess.run(f"ngrok config add-authtoken {authtoken}", shell=True, check=True)

def start_ngrok():
    ngrok_process = subprocess.Popen(["ngrok", "http", "5000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Wait for ngrok to initialize
    return ngrok_process

# Configure Stable Diffusion Pipeline (global initialization)
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")  # Move to GPU

API_KEY = "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"
MODEL_NAME = "qwen/qwen-vl-plus:free"
genai.configure(api_key="AIzaSyCcSn22t65ApHqRohShr7lefdbgS9icU2M")

# Configure SQLite Database for Chat History
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String, nullable=False)
    ai_response = db.Column(db.String, nullable=False)

with app.app_context():
    db.create_all()

# Summarizer Configuration (Gemini remains for Summarizer)
summarizer_generation_config = {
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}
summarizer_model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=summarizer_generation_config)

# Chatbot Function
def get_ai_response(user_input):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": user_input}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")

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
        user_message = request.form['message']
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
        summary_response = summarizer_model.generate_content(f"Summarize the following text: {user_text}")
        ai_summary = summary_response.text
        return jsonify(summary=ai_summary)
    return render_template('chatbot2.html')

# Updated Image Generator Route using Stable Diffusion
@app.route('/generate_image', methods=['POST'])
def generate_image():
    user_prompt = request.form['prompt']
    try:
        # Generate image using Stable Diffusion pipeline
        image = pipe(user_prompt).images[0]
        # Save the image to the static folder
        image_path = "./static/generated_image.png"
        if not os.path.exists('./static'):
            os.makedirs('./static')
        image.save(image_path)
        return jsonify(image_path=image_path)
    except Exception as e:
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
    app.run()      # Run Flask app with ngrok
