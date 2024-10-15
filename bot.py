from flask import Flask, render_template, request, jsonify, redirect
import google.generativeai as genai
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure the Google Generative AI client (use your own API key)
genai.configure(api_key="AIzaSyASPfaiJgcFLFuQB3aYz8iaH_pesNwQ6sM")  # Replace with your actual API key

# Set up the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define ChatHistory model
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String, nullable=False)
    ai_response = db.Column(db.String, nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

# Define generation configurations for chatbot and summarizer
chatbot_generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_output_tokens": 300,
    "response_mime_type": "text/plain",
}

summarizer_generation_config = {
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

# Initialize the chatbot and summarizer models
chatbot_model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=chatbot_generation_config)
summarizer_model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=summarizer_generation_config)

# Initialize chat session for chatbot
chat_session = chatbot_model.start_chat(history=[])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Chatbot route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form['message']
        response = chat_session.send_message(user_message)
        ai_response = response.text
        
        # Store in database
        new_chat = ChatHistory(user_message=user_message, ai_response=ai_response)
        db.session.add(new_chat)
        db.session.commit()
        
        return jsonify(response=ai_response)
    return render_template('chatbot1.html')

# Text summarizer route
@app.route('/summarizer', methods=['POST'])
def summarizer():
    user_text = request.form['text']
    summary_response = summarizer_model.generate_content(f"Summarize the following text: {user_text}")
    ai_summary = summary_response.text
    return jsonify(summary=ai_summary)

@app.route('/summarizer')
def summarizer_page():
    return render_template('chatbot2.html')

# Image generator route - Redirect to Gradio site
@app.route('/generate_image', methods=['POST', 'GET'])
def generate_image():
    # Redirect to the Gradio-hosted site for image generation
    return redirect("https://3b57b6fac52cf91dda.gradio.live/")

@app.route('/imagebot3')
def imagebot_page():
    return render_template('imagebot3.html')

# Route to get chat history
@app.route('/chat_history', methods=['GET'])
def chat_history():
    history = ChatHistory.query.all()
    return jsonify([{'user': chat.user_message, 'ai': chat.ai_response} for chat in history])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
