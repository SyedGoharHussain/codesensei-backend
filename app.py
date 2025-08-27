import os
from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from flask_cors import CORS # Import CORS
import datetime
import traceback

# Load environment variables
load_dotenv()

# --- Firebase Initialization ---
try:
    # Ensure the path to your service account key is correct
    cred = credentials.Certificate("codesensei-7591b-firebase-adminsdk-fbsvc-11e17cb667.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit()
genai.configure(api_key=GEMINI_API_KEY)
# Make model configurable via .env (for example: GEMINI_MODEL=gemini-2.5-flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
print(f"Using GEMINI_MODEL={GEMINI_MODEL}")

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_default_strong_secret_key")
# Enable CORS for your React app's origin
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])


# --- Socratic AI Prompts ---
CREATOR_NOTE = "Creator: Syed Gohar Hussain."
SYSTEM_PROMPTS = {
    "coding_coach": (
        CREATOR_NOTE + " " + (
            "You are CodeSensei, a 'Coding Coach'. Your goal is to help users learn by guiding them, "
            "not giving direct answers. Use the Socratic method. Ask probing questions that lead them "
            "to the solution. Provide small hints and conceptual explanations. Never write whole blocks of code. "
            "Your tone is encouraging, wise, and patient, like a sensei."
        )
    ),
    "debugging_assistant": (
        CREATOR_NOTE + " " + (
            "You are CodeSensei, a 'Debugging Assistant'. The user will provide code with errors. "
            "Analyze it carefully. Do not fix the code for them. Instead, identify the errors and give hints "
            "about where to look and what concepts might be involved. For example, say 'Look closely at your loop on line 5. "
            "What happens on the final iteration?' or 'That error often relates to variable types. Have you checked the type of 'x'?'"
        )
    ),
    "general": (
        CREATOR_NOTE + " " + (
            "You are CodeSensei, a helpful AI assistant with a Socratic teaching style. For any general question, "
            "your role is to foster understanding and critical thinking. Break down complex topics into smaller, "
            "manageable parts. Ask questions to gauge the user's understanding before providing more information. "
            "Guide them towards discovering the answer themselves."
        )
    )
}

# --- Helper function to verify Firebase ID token ---
def verify_token(request):
    try:
        id_token = request.headers.get('Authorization').split('Bearer ')[1]
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

# --- API Routes ---

@app.route("/api/get_chat_sessions")
def get_chat_sessions():
    """Fetches all chat session IDs for the current user."""
    user = verify_token(request)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        user_id = user['uid']
        sessions_ref = db.collection('users').document(user_id).collection('sessions').order_by('created_at', direction=firestore.Query.DESCENDING)
        sessions = [{"id": doc.id, "name": doc.to_dict().get("name", f"Session {doc.id[:5]}")} for doc in sessions_ref.stream()]
        return jsonify(sessions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/create_session", methods=["POST"])
def create_session():
    user = verify_token(request)
    if not user: return jsonify({"error": "Unauthorized"}), 401
    try:
        user_id = user['uid']
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        body = request.json or {}
        session_name = body.get('session_name')
        if session_name:
            # Use provided session_name as document ID (sanitized)
            doc_id = session_name
            new_session_ref = db.collection('users').document(user_id).collection('sessions').document(doc_id)
        else:
            new_session_ref = db.collection('users').document(user_id).collection('sessions').document()

        new_session_ref.set({
            "name": session_name or f"Chat - {timestamp}",
            "created_at": firestore.SERVER_TIMESTAMP
        })
        return jsonify({"session_id": new_session_ref.id, "name": session_name or f"Chat - {timestamp}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/get_session_messages/<session_id>")
def get_session_messages(session_id):
    """Fetches all messages for a specific session."""
    user = verify_token(request)
    if not user: return jsonify({"error": "Unauthorized"}), 401
    try:
        user_id = user['uid']
        messages_ref = db.collection('users').document(user_id).collection('sessions').document(session_id).collection('messages').order_by('timestamp')
        messages = [msg.to_dict() for msg in messages_ref.stream()]
        return jsonify(messages)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete_session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Deletes a chat session."""
    user = verify_token(request)
    if not user: return jsonify({"error": "Unauthorized"}), 401
    try:
        user_id = user['uid']
        # Note: Deleting a document does not delete its subcollections.
        # For a production app, use a Cloud Function to handle recursive deletes.
        db.collection('users').document(user_id).collection('sessions').document(session_id).delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Handles the chat interaction with the Gemini AI."""
    user = verify_token(request)
    if not user: return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    user_message = data.get("message")
    mode = data.get("mode", "general")
    session_id = data.get("session_id")
    user_id = user['uid']

    if not user_message or not session_id:
        return jsonify({"error": "Message and session_id are required"}), 400

    try:
        messages_collection = db.collection('users').document(user_id).collection('sessions').document(session_id).collection('messages')
        
        # Store user message
        messages_collection.add({
            'role': 'user',
            'content': user_message,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        # Fetch conversation history
        history_docs = messages_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20).stream()
        history = []
        for doc in history_docs:
            msg = doc.to_dict()
            role = "model" if msg.get('role') == 'assistant' else 'user'
            history.insert(0, {"role": role, "parts": [msg.get('content')]})


        # Generate AI Response
        model = genai.GenerativeModel(GEMINI_MODEL)
        system_instruction = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])

        conversation = [
            {"role": "user", "parts": [system_instruction]},
            {"role": "model", "parts": ["Understood. I will act as CodeSensei and guide the user."]}
        ] + history

        response = model.generate_content(conversation)
        # The Python client may return different shapes; try to recover text safely
        ai_message = getattr(response, 'text', None) or response
        if isinstance(ai_message, dict) and 'candidates' in ai_message:
            ai_message = ai_message['candidates'][0].get('content', '')

        # Store AI message
        messages_collection.add({
            'role': 'assistant',
            'content': ai_message,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return jsonify({"response": ai_message})

    except Exception as e:
        traceback.print_exc()
        print(f"Error during chat: {e}")
        return jsonify({"error": "An error occurred.", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001) # Running on a different port than React
