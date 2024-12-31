import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, DistilBertTokenizer, DistilBertModel
import torch

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Research Paper QA & Recommendation System", layout="wide")

# Load pre-trained QA model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='distilbert-base-uncased')

# Load pre-trained embedding model
@st.cache_resource
def load_bert_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

tokenizer, bert_model = load_bert_model()

# Load the dataset
@st.cache_data
def load_data():
    try:
        with open('models/data.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error("The dataset file 'data.pkl' was not found. Please ensure it is in the correct location.")
        return None

data = load_data()

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Function to answer a question based on the title and abstract or external input
def answer_question(query, context):
    try:
        result = qa_pipeline(question=query, context=context)
        return result['answer']
    except Exception as e:
        st.error(f"An error occurred during question answering: {e}")
        return None

# Function to check question relevance
def is_question_relevant(query, content, threshold=0.5):
    query_embedding = get_bert_embedding(query)
    content_embedding = get_bert_embedding(content)
    similarity = cosine_similarity(query_embedding.unsqueeze(0).numpy(), content_embedding.unsqueeze(0).numpy())[0][0]
    return similarity >= threshold

# Function to recommend papers based on keywords using TF-IDF and Cosine Similarity
def get_keyword_recommendations(query, data, top_n=5):
    # Combine title and abstract into a single string for TF-IDF vectorization
    combined_text = data['titles'] + ' ' + data['abstracts']
    
    # Initialize TF-IDF vectorizer and fit_transform the combined text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    # Transform the query using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity between the query and all papers
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)
    
    # Get the indices of the top N most similar papers
    similar_papers = cosine_sim[0].argsort()[-top_n-1:-1][::-1]
    
    return data.iloc[similar_papers][['titles', 'abstracts']]

# Streamlit UI
# Title and introductory text
st.title("Question Answering and Recommendation System for Research Papers")
st.write(
    "This system uses advanced Natural Language Processing techniques to help researchers ask questions about academic papers and get recommendations based on their queries. "
    "You can either ask questions based on selected research papers or get paper recommendations based on a keyword search."
)

# Sidebar layout for better navigation
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Choose the mode", options=["Select Mode", "Question Answer", "Recommendation"])

if mode == "Select Mode":
    # Show only the front page with the options
    st.write("Please select a mode to proceed.")
    
elif mode == "Question Answer":
    st.title("Question Answering System")

    if data is not None and not data.empty:
        # Select a paper from the dataset
        paper_index = st.selectbox("Select a Paper by Index", options=[None] + list(range(len(data))), format_func=lambda x: "Select a paper" if x is None else data['titles'].iloc[x])

        if paper_index is not None:
            # Display selected paper's title and abstract
            st.subheader("Selected Paper")
            st.write(f"**Title:** {data['titles'].iloc[paper_index]}")
            st.write(f"**Abstract:** {data['abstracts'].iloc[paper_index]}")

        # Input for external title and abstract
        external_title = st.text_area("Enter external paper title (optional):", height=68)
        external_abstract = st.text_area("Enter external paper abstract (optional):", height=200)

        # Input question
        query = st.text_input("Enter your question:")

        # Determine the context for answering the question
        if external_title.strip() or external_abstract.strip():
            context = external_title + ' ' + external_abstract
        elif paper_index is not None:
            context = data['titles'].iloc[paper_index] + ' ' + data['abstracts'].iloc[paper_index]
        else:
            context = None

        # Generate and display the answer
        if st.button("Get Answer"):
            if query.strip():
                if context:
                    with st.spinner("Processing your question..."):
                        answer = answer_question(query, context)
                        if answer:
                            st.success("Answer:")
                            st.write(answer)
                else:
                    st.error("Please select a paper or provide external title and abstract for context.")
            else:
                st.error("Please enter a valid question.")

elif mode == "Recommendation":
    st.header("Recommendation System")

    if data is not None and not data.empty:
        # Input for user's query
        query = st.text_input("Enter your query (e.g., keywords or topic):", placeholder="e.g., machine learning, deep learning")

        if query.strip():
            if st.button("Get Recommendations"):
                with st.spinner("Generating recommendations..."):
                    # Get keyword-based recommendations
                    recommended_papers = get_keyword_recommendations(query, data)
                    
                    # Display recommended papers
                    st.success("Recommended Papers:")
                    st.write(recommended_papers)
        else:
            st.warning("Please enter a query.")
    else:
        st.warning("Dataset is not loaded or is empty. Please check the file path and try again.")
else:
    st.warning("Please select a mode to proceed.")
