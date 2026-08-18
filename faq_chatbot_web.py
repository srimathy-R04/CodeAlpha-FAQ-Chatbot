"""
FAQ Chatbot — Streamlit Web UI
-----------------------------------------------------
A browser-based chat interface for the TF-IDF + Cosine Similarity FAQ bot.

Setup:
    pip install streamlit nltk scikit-learn
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

Run:
    streamlit run faq_chatbot_web.py

This opens a browser tab at http://localhost:8501 with a live chat window.
"""

import random
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. FAQ DATA — each entry can have several ways of asking the same thing
# ---------------------------------------------------------------------------
FAQ_DATA = [
    {
        "patterns": [
            "What are your business hours?",
            "When are you open?",
            "What time do you open and close?",
            "Are you open on weekends?",
        ],
        "answer": "We're open Monday to Friday, 9 AM to 6 PM EST. We're closed on weekends.",
    },
    {
        "patterns": [
            "How do I reset my password?",
            "I forgot my password",
            "Can't log in, password issue",
            "Change my password",
        ],
        "answer": "No worries — go to the login page, click 'Forgot Password', and "
                  "follow the instructions sent to your email.",
    },
    {
        "patterns": [
            "What payment methods do you accept?",
            "Can I pay with PayPal?",
            "Do you take credit cards?",
            "How can I pay for my order?",
        ],
        "answer": "We accept Visa, Mastercard, American Express, and PayPal.",
    },
    {
        "patterns": [
            "How can I track my order?",
            "Where is my package?",
            "Order tracking",
            "Has my order shipped yet?",
        ],
        "answer": "You can track your order using the link sent to your email after "
                  "checkout, or by logging into your account and checking Order History.",
    },
    {
        "patterns": [
            "What is your return policy?",
            "Can I return an item?",
            "Is returning allowed?",
            "How do refunds work?",
            "I want to return something",
        ],
        "answer": "You can return most items within 30 days of delivery for a full "
                  "refund, as long as they're unused and in original packaging.",
    },
    {
        "patterns": [
            "Do you offer international shipping?",
            "Do you ship outside the US?",
            "Can I get delivery to another country?",
        ],
        "answer": "Yes, we ship to over 50 countries. Shipping costs and delivery "
                  "times vary depending on your location.",
    },
    {
        "patterns": [
            "How do I contact customer support?",
            "I need help from a human",
            "Is there live chat?",
            "Customer service contact",
        ],
        "answer": "You can reach us anytime at support@example.com or through live "
                  "chat on our website — we're available 24/7.",
    },
    {
        "patterns": [
            "Can I cancel my subscription?",
            "How do I unsubscribe?",
            "Stop my membership",
        ],
        "answer": "Yes, you can cancel anytime from your account settings under "
                  "'Manage Subscription'. No cancellation fees apply.",
    },
    {
        "patterns": [
            "Is my personal data secure?",
            "Do you sell my data?",
            "Is my information safe with you?",
        ],
        "answer": "Yes — we use industry-standard encryption and never share your "
                  "data with third parties without your consent.",
    },
    {
        "patterns": [
            "Do you have a mobile app?",
            "Is there an app for this?",
            "Can I use this on my phone?",
        ],
        "answer": "Yes! Our app is available on both the App Store and Google Play.",
    },
]

SMALL_TALK = {
    "greeting": {
        "triggers": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "yo"],
        "responses": [
            "Hey there! 👋 What can I help you with today?",
            "Hi! How can I help?",
            "Hello! Ask me anything about your account, orders, or payments.",
        ],
    },
    "thanks": {
        "triggers": ["thanks", "thank you", "thank u", "thx", "appreciate it"],
        "responses": [
            "You're welcome! Anything else I can help with?",
            "Anytime! Let me know if you have more questions.",
            "Happy to help! 🙂",
        ],
    },
    "howareyou": {
        "triggers": ["how are you", "how r u", "how're you", "hows it going", "how is it going"],
        "responses": [
            "I'm doing well, thanks for asking! What can I help you with?",
            "All good here! What do you need help with today?",
        ],
    },
    "bye": {
        "triggers": ["bye", "goodbye", "see you", "later"],
        "responses": [
            "Take care! 👋",
            "Goodbye! Come back anytime.",
        ],
    },
}

FALLBACK_RESPONSES = [
    "Hmm, I'm not quite sure about that one. Try asking about orders, "
    "payments, returns, or your account.",
    "I don't have an answer for that yet. Could you rephrase, or ask about "
    "shipping, returns, or payments?",
    "Sorry, that's outside what I can help with right now. Try asking "
    "something about your order, account, or our policies.",
]


# ---------------------------------------------------------------------------
# 2. NLTK SETUP (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
    return True


ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in string.punctuation and tok not in stop_words
    ]
    return " ".join(cleaned)


def detect_small_talk(user_query: str):
    q = user_query.lower().strip(string.punctuation + " ")
    for category, data in SMALL_TALK.items():
        for trigger in data["triggers"]:
            if trigger in q:
                return category
    return None


# ---------------------------------------------------------------------------
# 3. CHATBOT CLASS
# ---------------------------------------------------------------------------
class FAQChatbot:
    def __init__(self, faq_data, similarity_threshold: float = 0.2):
        self.faq_data = faq_data
        self.similarity_threshold = similarity_threshold

        self.pattern_texts = []
        self.pattern_to_faq_index = []
        for faq_idx, item in enumerate(faq_data):
            for pattern in item["patterns"]:
                self.pattern_texts.append(pattern)
                self.pattern_to_faq_index.append(faq_idx)

        self.processed_patterns = [preprocess(p) for p in self.pattern_texts]

        self.word_vectorizer = TfidfVectorizer()
        self.word_vectors = self.word_vectorizer.fit_transform(self.processed_patterns)

        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.char_vectors = self.char_vectorizer.fit_transform(self.processed_patterns)

    def _best_match(self, user_query: str):
        processed_query = preprocess(user_query)
        if not processed_query.strip():
            return None, 0.0

        word_q = self.word_vectorizer.transform([processed_query])
        char_q = self.char_vectorizer.transform([user_query.lower()])

        word_sims = cosine_similarity(word_q, self.word_vectors).flatten()
        char_sims = cosine_similarity(char_q, self.char_vectors).flatten()

        combined = (0.75 * word_sims) + (0.25 * char_sims)

        best_pattern_idx = combined.argmax()
        best_score = combined[best_pattern_idx]
        best_faq_idx = self.pattern_to_faq_index[best_pattern_idx]
        return best_faq_idx, best_score

    def get_response(self, user_query: str):
        small_talk_category = detect_small_talk(user_query)
        if small_talk_category:
            response = random.choice(SMALL_TALK[small_talk_category]["responses"])
            return response, 1.0, f"[small talk: {small_talk_category}]"

        faq_idx, score = self._best_match(user_query)

        if faq_idx is None or score < self.similarity_threshold:
            return random.choice(FALLBACK_RESPONSES), score, None

        matched_pattern = self.pattern_texts[self.pattern_to_faq_index.index(faq_idx)]
        answer = self.faq_data[faq_idx]["answer"]
        return answer, score, matched_pattern

    def list_topics(self):
        return [item["patterns"][0] for item in self.faq_data]


@st.cache_resource
def load_bot():
    return FAQChatbot(FAQ_DATA, similarity_threshold=0.2)


# ---------------------------------------------------------------------------
# 4. STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.title("💬 FAQ Chatbot")
st.caption("Built with NLTK + TF-IDF + Cosine Similarity")

bot = load_bot()

# Sidebar: topics + settings
with st.sidebar:
    st.header("What I can help with")
    for topic in bot.list_topics():
        st.markdown(f"- {topic}")

    st.divider()
    debug_mode = st.checkbox("Show match confidence", value=False)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 Ask me anything about orders, payments, returns, or your account."}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if debug_mode and msg.get("debug"):
            st.caption(msg["debug"])

# Chat input
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer, score, matched = bot.get_response(user_input)
    debug_text = f"matched: \"{matched}\" | confidence: {score:.2f}" if matched else f"no match | confidence: {score:.2f}"

    with st.chat_message("assistant"):
        st.markdown(answer)
        if debug_mode:
            st.caption(debug_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "debug": debug_text}
    )
