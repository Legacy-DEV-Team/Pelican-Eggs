import os
import subprocess
import sys
import sqlite3
import ssl
import torch
import eventlet
import eventlet.wsgi
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define AI Assistant directories
AI_DIR = "/home/container"
DB_PATH = f"{AI_DIR}/ai_assistant.db"
SYSTEMD_SERVICE = "/etc/systemd/system/ai_auth.service"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
socketio = SocketIO(app, cors_allowed_origins="*")

# Database Setup
db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

def install_dependencies():
    """Installs required system and Python dependencies."""
    print("🚀 Installing dependencies...")
    subprocess.run(["apt", "update"], check=True)
    subprocess.run(["apt", "install", "-y", "curl", "git", "python3-pip", "python3-venv", "sqlite3"], check=True)

    os.makedirs(AI_DIR, exist_ok=True)

    subprocess.run(["python3", "-m", "venv", f"{AI_DIR}/venv"], check=True)
    subprocess.run([f"{AI_DIR}/venv/bin/pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([f"{AI_DIR}/venv/bin/pip", "install", "flask", "flask_sqlalchemy", "flask_socketio", "eventlet", "torch", "transformers"], check=True)

def setup_systemd_service():
    """Creates and enables a systemd service for the AI Assistant."""
    print("🛠️ Setting up systemd service...")
    service_content = f"""
[Unit]
Description=AI Assistant WebSocket Service
After=network.target

[Service]
User=root
WorkingDirectory={AI_DIR}
ExecStart={AI_DIR}/venv/bin/python {AI_DIR}/aistart.py
Restart=always

[Install]
WantedBy=multi-user.target
"""
    with open(SYSTEMD_SERVICE, "w") as f:
        f.write(service_content)

    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "ai_auth.service"], check=True)
    subprocess.run(["systemctl", "start", "ai_auth.service"], check=True)

# Load AI Model
MODEL_NAME = "facebook/opt-1.3b"  # Change this to any supported model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("user_message")
def handle_chat(data):
    """Handles user messages & sends live AI responses via WebSockets"""
    username = data["username"]
    user_message = data["message"]

    chat_history = f"User: {user_message}\nAI:"
    input_ids = tokenizer.encode(chat_history, return_tensors="pt").to("cuda")

    response_text = ""
    for token in model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id, do_sample=True):
        response_text = tokenizer.decode(token, skip_special_tokens=True)
        emit("ai_response", {"message": response_text}, broadcast=True)

    # Store chat in database
    new_chat = ChatHistory(username=username, message=user_message, response=response_text)
    db.session.add(new_chat)
    db.session.commit()

def start_ai():
    """Main function to install, configure, and start AI Assistant."""
    install_dependencies()
    setup_systemd_service()
    print("✅ AI Assistant is running at https://ai.core-x.dev & WebSockets at wss://aiip.core-x.dev!")

if __name__ == "__main__":
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain("/etc/letsencrypt/live/aiip.core-x.dev/fullchain.pem", "/etc/letsencrypt/live/aiip.core-x.dev/privkey.pem")

    socketio.run(app, host="0.0.0.0", port=443, ssl_context=context)
