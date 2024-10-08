
# AI Mental Health Assistant

This project is an AI-based Mental Health Assistant that helps users with mental health support using natural language processing and AI.

## Features:
- Conversational interface to assist users with mental health concerns.
- Provides mental health advice and information.
- Uses machine learning for improved responses.

## Installation:

1. Clone this repository:
   ```bash
   git clone https://github.com/<username>/mental-health-assistant.git
   cd mental-health-assistant
# Frontend code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice-Enabled Chatbot</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(to right, #667eea, #764ba2);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }

        .chat-container {
            width: 400px;
            background-color: #fff;
            border-radius: 15px;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background-color: #764ba2;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            position: relative;
        }

        .chat-header i {
            position: absolute;
            right: 20px;
            top: 18px;
        }

        .chat-box {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
            background-color: #f9f9f9;
        }

        .message {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            line-height: 1.4;
            word-wrap: break-word;
            position: relative;
            animation: fadeIn 0.3s ease-in-out;
        }

        .message.user {
            background-color: #667eea;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 5px;
        }

        .message.bot {
            background-color: #e1e1e1;
            color: #333;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
        }

        .input-container {
            display: flex;
            padding: 15px;
            background-color: #f9f9f9;
            border-top: 1px solid #ddd;
        }

        .input-container input {
            flex: 1;
            padding: 10px;
            font-size: 1rem;
            border: 1px solid #ddd;
            border-radius: 20px;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-container input:focus {
            border-color: #764ba2;
        }

        .input-container button {
            padding: 10px 20px;
            font-size: 1rem;
            background-color: #764ba2;
            color: white;
            border: none;
            border-radius: 20px;
            margin-left: 10px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .input-container button:hover {
            background-color: #5e388d;
        }

        .mic-button {
            background-color: #667eea;
            color: white;
            border: none;
            border-radius: 50%;
            font-size: 1.5rem;
            padding: 10px;
            cursor: pointer;
            margin-left: 10px;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Chat bubbles arrows */
        .message.bot::before {
            content: "";
            position: absolute;
            top: 0;
            left: -8px;
            border-width: 8px;
            border-style: solid;
            border-color: transparent transparent transparent #e1e1e1;
        }

        .message.user::before {
            content: "";
            position: absolute;
            top: 0;
            right: -8px;
            border-width: 8px;
            border-style: solid;
            border-color: transparent #667eea transparent transparent;
        }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="chat-header">
        <span>Voice Chatbot Assistant</span>
        <i class="fas fa-robot"></i>
    </div>
    <div class="chat-box" id="chat-box">
        <!-- Chat messages will appear here -->
    </div>
    <div class="input-container">
        <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off">
        <button onclick="sendMessage()">Send <i class="fas fa-paper-plane"></i></button>
        <button class="mic-button" onclick="startRecognition()"><i class="fas fa-microphone"></i></button>
    </div>
</div>

<script>
    // Function to add messages to the chatbox
    function addMessageToChatBox(message, isUser) {
        const chatBox = document.getElementById('chat-box');
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user' : 'bot');
        messageElement.textContent = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Text-to-Speech function with female voice
    function speakMessage(message) {
        const utterance = new SpeechSynthesisUtterance(message);
        
        // Get available voices and select a female voice
        const voices = window.speechSynthesis.getVoices();
        const femaleVoice = voices.find(voice => voice.name.includes('Google UK English Female') || voice.name.includes('Female') || voice.name.includes('en-US') && voice.gender === "female");

        // Set the selected female voice (if available)
        if (femaleVoice) {
            utterance.voice = femaleVoice;
        } else {
            console.log("Female voice not found, using default voice.");
        }
        
        window.speechSynthesis.speak(utterance);
    }

    // Function to send message
    async function sendMessage() {
        const userInputElement = document.getElementById('user-input');
        const userInput = userInputElement.value;
        if (userInput.trim() === '') return;

        // Add user message to the chatbox
        addMessageToChatBox(userInput, true);

        // Clear input field
        userInputElement.value = '';

        // Send user message to the server
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: userInput }),
        });

        const data = await response.json();

        // Add bot response to the chatbox and speak it with a female voice
        addMessageToChatBox(data.response, false);
        speakMessage(data.response); // Bot speaks the response
    }

    // Allow 'Enter' key to send the message
    document.getElementById('user-input').addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Speech Recognition setup
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('user-input').value = transcript;
        sendMessage(); // Automatically send the recognized message
    };

    function startRecognition() {
        recognition.start();
    }
</script>


</body>
</html>
