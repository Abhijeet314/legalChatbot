from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# In-memory conversation storage
# In a production environment, consider using a database
conversation_history = {}

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("query")
        conversation_id = data.get("conversationId", "default")
        
        if not user_query:
            return jsonify({"error": "Query is required"}), 400
            
        print(f"Received query: {user_query} for conversation: {conversation_id}")
        
        # Initialize conversation history if it doesn't exist
        if conversation_id not in conversation_history:
            system_prompt = """
            You are Advocate.ai, a sophisticated legal assistant specialized in all aspects of law.
            You provide accurate, nuanced responses to legal queries based on comprehensive knowledge of:
            
            - All legal codes, statutes, and regulations from major jurisdictions
            - Case law and legal precedents across different courts
            - Civil, criminal, corporate, constitutional, and administrative law
            - Legal procedures, court protocols, and filing requirements
            - Contract law and document analysis
            
            Guidelines for responses:
            - Always provide balanced legal perspectives considering multiple interpretations
            - Cite relevant sections of law, case precedents, and statutes when appropriate
            - Explain complex legal concepts in clear, accessible language
            - When appropriate, outline potential strategies or approaches a legal professional might consider
            - Always clarify that you're providing legal information, not legal advice
            - Acknowledge jurisdictional differences when relevant
            
            When uncertain, acknowledge limitations and suggest consulting with a qualified attorney for specific advice.
            """
            
            # Start a chat session with Gemini
            chat = model.start_chat(history=[])
            conversation_history[conversation_id] = {
                "chat": chat,
                "system_prompt": system_prompt
            }
        
        # Get the chat session
        chat = conversation_history[conversation_id]["chat"]
        system_prompt = conversation_history[conversation_id]["system_prompt"]
        
        # Prepend system prompt to first message or send as context
        if len(chat.history) == 0:
            full_query = f"{system_prompt}\n\nUser: {user_query}"
        else:
            full_query = user_query
        
        # Send message and get response
        response = chat.send_message(full_query)
        response_content = response.text
        
        print(f"LLM response: {response_content}")
        
        return jsonify({
            "response": response_content,
            "conversationId": conversation_id
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        data = request.get_json()
        conversation_id = data.get("conversationId", "default")
        
        # Reset conversation by creating a new chat session
        if conversation_id in conversation_history:
            system_prompt = conversation_history[conversation_id]["system_prompt"]
            chat = model.start_chat(history=[])
            conversation_history[conversation_id] = {
                "chat": chat,
                "system_prompt": system_prompt
            }
        
        return jsonify({"status": "success", "message": "Conversation history cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simple test endpoint that doesn't use LLM
@app.route("/test", methods=["POST"])
def test():
    data = request.get_json()
    query = data.get("query", "no query")
    return jsonify({"response": f"Echo: {query}"})

if __name__ == "__main__":
    app.run(debug=True)