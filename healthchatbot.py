import nltk
import streamlit as st
import re
import requests
import json
import string
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Download required nltk data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load stop words
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer globally
lemmatizer = WordNetLemmatizer()

# 🔥 Define preprocess function OUTSIDE of the if-statement
def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Load the text file
file_path = r"C:\Users\Pearl\OneDrive\Documents\GOMYCODE TRAINING\Machine_learning\exercise_wellness text.txt"  # Ensure the file exists
try:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    st.error(f"Error: The file '{file_path}' was not found. Ensure it is in the correct directory.")
    text = ""

# Data Cleaning & Processing
if text:
    text = re.sub(r"[^a-zA-Z\s.!?]", "", text)  # Keep punctuation for sentence splitting
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = re.sub(r"\n\d+\n", "\n", text)  # Remove page numbers
    #text = text.lower()  # Convert to lowercase

    # Split text into sentences
    
    original_sentences = sent_tokenize(text)

    # Preprocess all sentences
    processed_sentences = [preprocess(sentence) for sentence in original_sentences]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

def format_response(sentences):  
    """  
    Takes a list of sentences and returns a structured, natural-sounding response.  
    """  
    if not sentences:  
        return "🤔 Sorry, I don’t have an answer for that."  

    # Capitalize the first letter of the first sentence  
    structured_response = " .".join(sentences) 
    structured_response = re.sub(r'\s+([.!?])', r'\1', structured_response)  # Remove spaces before punctuation  
    structured_response = structured_response[0].capitalize() + structured_response[1:]  # Capitalize first letter of response  

    url = "https://textgears-textgears-v1.p.rapidapi.com/correct"

    payload = { "text": structured_response }
    headers = {
	"x-rapidapi-key": "13f096eaeamsh03073a9ee35b235p198a05jsn6dbdf9fc99ed",
	"x-rapidapi-host": "textgears-textgears-v1.p.rapidapi.com",
	"Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        o =  response.json()
        formatted_response =  o["response"]["corrected"]
        text = formatted_response
        blob = TextBlob(text)
        corrected_text = str(blob.correct())
        return corrected_text
        
    else:
        return structured_response


# Function to find relevant sentences
def get_most_relevant_sentences(query):
    if not query:
        return "⚠️ Please ask a question."

    query_processed = preprocess(query)
    query_vector = vectorizer.transform([query_processed])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    # Get top matches above threshold
    threshold = max(0.2, similarity_scores.max() * 0.6)  # Ensure meaningful matches
    best_sentences = [original_sentences[i] for i in similarity_scores.argsort()[::-1] if similarity_scores[i] >= threshold]

    # 🔥 Return structured response
    return format_response(best_sentences)

# Chatbot function
def chatbot(question):
    return get_most_relevant_sentences(question)


# Streamlit UI
def main():
    st.set_page_config(page_title="Exercise & Wellness Chatbot 🏋️‍♂️", page_icon="💪", layout="centered")
    st.sidebar.title("⚙️ Settings")
    st.sidebar.write("Enhance your experience with these options:")

    # 🔥 Ensure chat history is initialized at the beginning
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

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

    # Use predefined question if selected
    if selected_question != "Select a question...":
        question = selected_question

    # Submit Button
    if st.button("🚀 Ask"):
        if question.strip():
            with st.spinner("Thinking... 🤖"):
                time.sleep(1)
                response = chatbot(question)

                # 🔥 Append to chat history
                st.session_state.chat_history.append(("🧑‍💻 You", question))
                st.session_state.chat_history.append(("🤖 Chatbot", response))

                st.write(f"**🤖 Chatbot:** {response}")
        else:
            st.warning("⚠️ Please enter a question.")

    # Display Chat History
    if st.session_state["chat_history"]:
        st.markdown("### 📜 Conversation History")
        for speaker, msg in st.session_state["chat_history"]:
            st.write(f"**{speaker}:** {msg}")

    # Clear Chat Button
    if st.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []  # Reset history
        st.success("Chat history cleared!")

if __name__ == "__main__":
    main()
