import streamlit as st
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer

# Page configuration
st.set_page_config(
    page_title="AI Chatbot | Intelligent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    
    /* Messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #1f1f1f;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 70%;
        margin-right: auto;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 20px;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Input */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-block;
        padding: 10px;
    }
    
    .typing-indicator span {
        height: 10px;
        width: 10px;
        background: #667eea;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    /* Stats */
    .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


class ChatbotEngine:
    """Chatbot inference engine."""
    
    def __init__(self, model_path: str = 'src/chatbot_model.pkl'):
        """Load trained model."""
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components."""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.intents = data['intents']
            self.stemmer = data['stemmer']
            self.stop_words = data['stop_words']
            self.metadata = data.get('metadata', {})
            
            return True
        except FileNotFoundError:
            st.error("⚠️ Model not found! Please train the model first.")
            st.info("Run: `python src/train_model.py`")
            return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess user input."""
        try:
            tokens = nltk.word_tokenize(text.lower())
        except:
            tokens = text.lower().split()
        
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def predict(self, user_input: str):
        """Predict intent and get response."""
        # Preprocess
        processed = self.preprocess_text(user_input)
        
        # Vectorize
        features = self.vectorizer.transform([processed])
        
        # Predict
        intent = self.model.predict(features)[0]
        
        # Get confidence
        try:
            proba = self.model.predict_proba(features)[0]
            confidence = max(proba)
        except:
            confidence = 1.0
        
        # Get response
        for intent_data in self.intents:
            if intent_data['tag'] == intent:
                response = random.choice(intent_data['responses'])
                return response, intent, confidence
        
        return "I'm not sure how to respond to that. Can you rephrase?", "unknown", 0.0


def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_count' not in st.session_state:
        st.session_state.chat_count = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now()
    if 'bot_loaded' not in st.session_state:
        st.session_state.bot_loaded = False
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None


def load_chatbot():
    """Load chatbot instance."""
    if not st.session_state.bot_loaded:
        with st.spinner("🔄 Loading AI model..."):
            chatbot = ChatbotEngine()
            if chatbot.model:
                st.session_state.chatbot = chatbot
                st.session_state.bot_loaded = True
                return True
            return False
    return True


def main():
    """Main application."""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Intelligent Chatbot</h1>
        <p>Your 24/7 Virtual Assistant | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load chatbot
    if not load_chatbot():
        st.stop()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{st.session_state.chat_count}</div>
                <div class="stat-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            uptime = (datetime.now() - st.session_state.start_time).seconds // 60
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{uptime}m</div>
                <div class="stat-label">Session</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Info
        st.markdown("### 🧠 Model Information")
        if hasattr(chatbot, 'metadata'):
            metadata = chatbot.metadata
            st.info(f"""
            **Model:** {metadata.get('model_type', 'N/A')}  
            **Accuracy:** {metadata.get('accuracy', 0)*100:.1f}%  
            **Intents:** {metadata.get('num_intents', 0)}  
            **Trained:** {metadata.get('training_date', 'N/A')}
            """)
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_count = 0
            st.rerun()
        
        if st.button("📊 View Stats", use_container_width=True):
            st.info(f"Total messages: {st.session_state.chat_count}")
        
        st.markdown("---")
        
        # Sample Questions
        st.markdown("### 💡 Try Asking")
        sample_questions = [
            "Hello!",
            "What can you do?",
            "Tell me a joke",
            "Who created you?",
            "What's your name?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.user_input = q
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 20px;'>
            <p>Built with ❤️ using Streamlit</p>
            <p>Powered by Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main chat area
    st.markdown("### 💬 Chat with AI Assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style='text-align: right;'>
                    <div class='user-message'>
                        <strong>You:</strong><br>
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = message.get('confidence', 0)
                intent = message.get('intent', 'unknown')
                
                st.markdown(f"""
                <div style='text-align: left;'>
                    <div class='bot-message'>
                        <strong>🤖 AI Assistant:</strong><br>
                        {message['content']}
                        <div style='font-size: 0.8em; color: #666; margin-top: 8px;'>
                            Intent: {intent} | Confidence: {confidence*100:.0f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            key="user_input_field",
            placeholder="Ask me anything! 💭",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 🚀", use_container_width=True)
    
    # Handle input
    if send_button or (user_input and user_input != st.session_state.get('last_input', '')):
        if user_input and user_input.strip():
            st.session_state.last_input = user_input
            
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Show typing indicator
            with st.spinner("🤔 Thinking..."):
                time.sleep(0.5)  # Simulate thinking
                
                # Get bot response
                response, intent, confidence = chatbot.predict(user_input)
                
                # Add bot message
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response,
                    'intent': intent,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
                
                st.session_state.chat_count += 2
            
            # Rerun to update display
            st.rerun()
    
    # Welcome message
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class='info-box' style='text-align: center; padding: 40px;'>
            <h3>👋 Welcome to AI Chatbot!</h3>
            <p>I'm your intelligent virtual assistant powered by machine learning.</p>
            <p><strong>Try asking me:</strong></p>
            <ul style='list-style: none; padding: 0;'>
                <li>❓ Questions about anything</li>
                <li>💡 For recommendations and advice</li>
                <li>🎭 To tell you a joke</li>
                <li>🤖 About my capabilities</li>
            </ul>
            <p style='margin-top: 20px;'><em>Type a message below to get started!</em></p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
