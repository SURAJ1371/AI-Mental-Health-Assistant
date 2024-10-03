from flask import Flask, render_template, request, jsonify
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2
from deepface import DeepFace

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('chatbot_data.csv')

# Load the pretrained model and tokenizer from Huggingface
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a dictionary from the CSV for lookup
response_dict = dict(zip(df['user_input'], df['assistant_response']))

# Method to get model-based response if exact match isn't found
def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# OpenCV method to detect emotion from the webcam feed
def detect_emotion():
    cap = cv2.VideoCapture(0)  # Access webcam
    emotion = "neutral"  # Default emotion

    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break

        # Use DeepFace to analyze the emotion
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result['dominant_emotion']
        except:
            pass

        # Display the webcam feed with the detected emotion
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam - Press Q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion

# Route for the chatbot interaction
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    # Exact match lookup in dataset
    if user_input in response_dict:
        response = response_dict[user_input]
    else:
        # Use the model if not in dataset
        response = generate_response(user_input)
    
    # Run emotion detection
    detected_emotion = detect_emotion()
    response += f" (Detected emotion: {detected_emotion})"
    
    return jsonify({"response": response})

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
