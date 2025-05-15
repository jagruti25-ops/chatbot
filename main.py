import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------------------
# Load Embedding Model
# ----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------
# Sample Teachings (can be loaded from a file)
# ----------------------
teachings = [
    "Kabir says: Listen carefully, my friend, love is the only reality.",
    "Ramana Maharshi teaches that the true Self is beyond the mind.",
    "Sai Baba says: Shraddha (faith) and Saburi (patience) are the essence of life.",
    "Swami Vivekananda: Arise, awake, and stop not till the goal is reached."
]

# ----------------------
# Embed the Teachings
# ----------------------
embeddings = model.encode(teachings)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Spiritual Chatbot", page_icon="🧘")
st.title("🧘 How can we assist on your spiritual journey today ")
st.markdown("Ask a spiritual question and receive answers based on saints' wisdom.")

user_query = st.text_input("Your Question:", placeholder="e.g., What is true love?")

if user_query:
    query_vec = model.encode([user_query])
    D, I = index.search(np.array(query_vec), k=1)
    top_idx = I[0][0]
    top_response = teachings[top_idx]

    st.markdown("---")
    st.subheader("🕉️ Answer:")
    st.write(top_response)
    st.markdown("---")
    st.caption("Response based on teachings stored locally.")