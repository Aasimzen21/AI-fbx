import os
import requests
from flask import Flask, render_template, request, jsonify, redirect
import google.generativeai as genai
from flask_sqlalchemy import SQLAlchemy
from PyPDF2 import PdfReader
import json

app = Flask(__name__)

# ✅ Configure LLaMA 3.2 Chatbot
API_KEY = "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"  # Replace with valid OpenRouter API key
MODEL_NAME = "qwen/qwen-vl-plus:free"
genai.configure(api_key="AIzaSyCcSn22t65ApHqRohShr7lefdbgS9icU2M")

# ✅ Configure SQLite Database for Chat History
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String, nullable=False)
    ai_response = db.Column(db.String, nullable=False)

with app.app_context():
    db.create_all()

# ✅ Summarizer Configuration (Gemini remains for Summarizer)
summarizer_generation_config = {
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}
summarizer_model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=summarizer_generation_config)

# ✅  Chatbot Function
def get_ai_response(user_input):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": user_input}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")

# ✅ Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ✅ LLaMA Chatbot Route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form['message']
        ai_response = get_ai_response(user_message)
        
        # Store in database
        new_chat = ChatHistory(user_message=user_message, ai_response=ai_response)
        db.session.add(new_chat)
        db.session.commit()
        
        return jsonify(response=ai_response)
    return render_template('chatbot1.html')

# ✅ Summarizer Route (No Changes)
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

# ✅ Image Generator Route (No Changes)
@app.route('/generate_image', methods=['POST'])
def generate_image():
    user_prompt = request.form['prompt']
    api_key = "sk-jv0htZnscjoIgaSxVqmOsmDFyyvHG1G2wLfjYudpkssts7rv"  # Replace with actual API key
    
    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/ultra",
        headers={
            "authorization": f"Bearer {api_key}",
            "accept": "image/*"
        },
        files={"none": ''},
        data={"prompt": user_prompt, "output_format": "png"},
    )

    if response.status_code == 200:
        image_path = "./static/lighthouse.webp"
        with open(image_path, 'wb') as file:
            file.write(response.content)
        return jsonify(image_path=image_path)
    else:
        return jsonify(error=str(response.json()))

@app.route('/imagebot3')
def imagebot_page():
    return render_template('imagebot3.html')

# ✅ Chat History Route
@app.route('/chat_history', methods=['GET'])
def chat_history():
    history = ChatHistory.query.all()
    return jsonify([{'user': chat.user_message, 'ai': chat.ai_response} for chat in history])

# ✅ Clear Chat Route
@app.route('/clear_chat', methods=['GET'])
def clear_chat():
    ChatHistory.query.delete()
    db.session.commit()
    return jsonify({"status": "success"})

# ✅ Delete Chat History Route
@app.route('/delete_history', methods=['GET'])
def delete_history():
    ChatHistory.query.delete()
    db.session.commit()
    return jsonify({"status": "deleted"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
