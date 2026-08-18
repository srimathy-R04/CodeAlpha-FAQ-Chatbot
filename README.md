## 💬 FAQ Chatbot using NLP

# An AI-powered FAQ chatbot developed as part of the CodeAlpha Artificial Intelligence Internship.

# The chatbot uses Natural Language Processing (NLP) techniques to understand user questions and find the most relevant answer from a predefined FAQ dataset.

## 🚀 Features

- Interactive web-based chatbot using Streamlit
- FAQ dataset containing multiple questions and answers
- Text preprocessing using NLTK
- Tokenization
- Stopword removal
- Lemmatization
- TF-IDF vectorization
- Cosine similarity for question matching
- Character-level TF-IDF for additional matching support
- Small-talk handling for greetings, thanks, and goodbye messages
- Fallback responses for questions outside the available FAQ topics
- Optional match-confidence display
- Chat history using Streamlit session state

## 🧠 Technologies Used

- Python
- Streamlit
- NLTK
- Scikit-learn
- NumPy
- SciPy

## 🔄 How It Works

# The chatbot follows this workflow:

User Question
↓
Text Preprocessing using NLTK
↓
TF-IDF Vectorization
↓
Cosine Similarity Calculation
↓
Find Most Similar FAQ
↓
Return the Corresponding Answer

## 📚 FAQ Topics

# The chatbot currently handles questions related to:

- Business hours
- Password reset
- Payment methods
- Order tracking
- Return policy
- International shipping
- Customer support
- Subscription cancellation
- Data security
- Mobile application

## 🖥️ Running the Project Locally

1. Clone the repository

git clone https://github.com/srimathy-R04/CodeAlpha-FAQ-Chatbot

2. Open the project folder

cd CodeAlpha-FAQ-Chatbot

3. Install the required packages

pip install -r requirements.txt

4. Download the required NLTK resources

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

5. Run the Streamlit application

streamlit run faq_chatbot_web.py

The application will open in your browser at the local Streamlit address.

## 📸 Screenshots

# Chatbot Interface

"FAQ Chatbot" (Chatbot Interface.png)

# Password Reset Question

"Password Reset" (Password Reset.png)

# Order Tracking Question

"Order Tracking" (Order Tracking.png)

## 🎯 Internship Task

Organization: CodeAlpha
Task: Task 2 – Chatbot for FAQs
Domain: Artificial Intelligence / Natural Language Processing

## 👩‍💻 Author

# Srimathy R

# B.Tech Computer Science and Engineering Student

## 📌 Project Objective

The objective of this project is to develop an FAQ chatbot capable of matching user questions with predefined frequently asked questions using NLP preprocessing, TF-IDF vectorization, and cosine similarity.

## 📄 License

This project was developed for educational and internship purposes.
