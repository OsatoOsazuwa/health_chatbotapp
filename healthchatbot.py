import nltk
import streamlit as st
import re
import string
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required nltk data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Load stop words
stop_words = set(stopwords.words('english'))

# Load the text file
file_path = r"C:\Users\Pearl\Downloads\exercise_wellness text.txt"
try:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    st.error(f"Error: The file at {file_path} was not found.")
    text = ""

# Data Cleaning
if text:
    text = re.sub(r"[^a-zA-Z\s.!?]", "", text)  # Keep . ! ? for sentence splitting
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = re.sub(r"\n\d+\n", "\n", text)  # Remove page numbers
    text = text.lower()  # Convert to lowercase

    # Split text into sentences
    original_sentences = re.split(r'(?<=[.!?])\s+', text)
    original_sentences = [sent for sent in original_sentences if len(sent.split()) >= 5]

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Preprocessing function
    def preprocess(sentence):
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    # Preprocess all sentences
    processed_sentences = [preprocess(sentence) for sentence in original_sentences]
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # Function to find relevant sentences
    def get_most_relevant_sentences(query):
        query_processed = preprocess(query)
        query_vector = vectorizer.transform([query_processed])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

        # Get top matches above threshold
        threshold = max(0.2, similarity_scores.max() * 0.7)
        best_sentences = [original_sentences[i] for i in similarity_scores.argsort()[::-1] if similarity_scores[i] >= threshold]

        return " ".join(best_sentences[:2]) if best_sentences else "🤔 Sorry, I don’t have an answer for that."

    # Chatbot function
    def chatbot(question):
        return get_most_relevant_sentences(question)

    # Streamlit UI
    def main():
        st.set_page_config(page_title="Exercise & Wellness Chatbot 🏋️‍♂️", page_icon="💪", layout="centered")
        st.sidebar.title("⚙️ Settings")
        st.sidebar.write("Enhance your experience with these options:")
    
    # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🏋️ Exercise & Wellness Chatbot 💪</h1>", unsafe_allow_html=True)
        st.write("Hello! I'm a chatbot. Ask me anything about fitness, exercise, and wellness! 🏃‍♀️")
    
    # Predefined Questions Dropdown
        predefined_questions = [
            "What are the benefits of regular exercise?",
            "How can I improve my flexibility?",
            "What is the best diet for muscle gain?",
            "How often should I work out?",
            "What are the key principles of wellness?"
            ]
        selected_question = st.selectbox("🔍 Choose a question:", ["Select a question..."] + predefined_questions)
    
    # User Input
        question = st.text_input("💬 You:", placeholder="Type your question here...")
    
        if selected_question != "Select a question...":
            question = selected_question
    
    # Submit Button
        if st.button("🚀 Ask"):
            if question.strip():
                with st.spinner("Thinking... 🤖"):
                    time.sleep(1)
                    response = chatbot(question)
                    st.session_state.chat_history.append(("🧑‍💻 You", question))
                    st.session_state.chat_history.append(("🤖 Chatbot", response))
                    st.write(f"**🤖 Chatbot:** {response}")
            else:
                st.warning("⚠️ Please enter a question.")
    
    # Display Chat History
        if st.session_state.chat_history:
            st.markdown("### 📜 Conversation History")
            for speaker, msg in st.session_state.chat_history:
                st.write(f"**{speaker}:** {msg}")
    
    # Clear Chat Button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

if __name__ == "__main__":
        main()