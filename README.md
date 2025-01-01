# Research Paper Question Answering & Recommendation System

This project provides a Question Answering and Recommendation System for research papers using DistilBERT and Natural Language Processing (NLP). Users can ask questions about research papers and receive answers, as well as get paper recommendations based on their input.

# 1. Dataset Preparation
* Data Import: The dataset is loaded from a file containing research paper abstracts, titles, and terms.
* Sampling: Only 20% of the data is used for faster processing during development.
- **Preprocessing:**
  - Handle missing values and duplicates.
    - Fill missing abstracts with an empty string.
    - Drop rows with missing titles or terms.
  - Rare terms are filtered out to focus on frequently occurring topics.
    - Keep terms that appear at least 5 times.
  - Conversion of string representations of lists into Python list objects using `literal_eval`.

# 2. Visualization

**Term Frequency Distribution:**
- Top 20 most frequent terms before and after filtering are plotted.
**Embedding Visualization:**
- PCA is used to reduce combined embeddings to 2D for scatter plot visualization, showcasing the relative positioning of papers in the embedding space.
# 3. Feature Extraction with Transformers
**BERT Embeddings:**
- Titles, abstracts, and terms are converted into vector embeddings using **DistilBERT**.
- A helper function processes the data in batches to handle memory constraints effectively.
- Title, abstract, and term embeddings are concatenated to form a combined embedding representing the research paper.
# 4. Similarity Calculation
**Cosine Similarity:**
- A similarity matrix is computed for all paper embeddings.
- Heatmaps visualize similarity scores between subsets of papers, aiding exploratory analysis.
# 5. Recommendation System
**Top-N Recommendations:**
- For a given paper, cosine similarity identifies the most relevant papers.
- Titles and terms of the top-N recommended papers are displayed, which can help users find related work.
# 6. Question Answering System
**QA Pipeline:**
- Uses `distilbert-base-uncased-distilled-squad` to answer questions based on a paper’s title and abstract.
- Questions can also be answered for new papers (not in the dataset) by processing their titles and abstracts.
# 7. Streamlit Integration

**Modes:**
- Question Answer: Allows users to query based on specific papers or external input.
- Recommendation: Suggests papers related to a user's query based on keywords or topics.

**Interactivity:**
- Users can select a paper from a dropdown, enter a custom query, or provide external paper details for analysis.
# 8. Future Enhancements
**Embedding Compression:** Use dimensionality reduction (e.g., PCA or TSNE) to speed up similarity calculations.

**Hybrid Recommendation:** Combine keyword-based and embedding-based methods for more robust suggestions.

**Dataset Management:** Add functionality to upload custom datasets directly via Streamlit.

# Tech Stack
* Python
* Streamlit
* DistilBERT (Hugging Face)
* Scikit-learn
* PyTorch
* Pandas
* Seaborn
* Matplotlib

# Installation
1. ### Clone the repository:
git clone 
https://github.com/Vaibhavipowar2023/Research-Paper-Recommendation-and-Question-Answer-System-using-DistilBERT.git

cd Research-Paper-Recommendation-and-Question-Answer-System-using-DistilBERT

2. ### Dataset:
Make sure you download the Arxiv Paper Abstracts dataset from this link: [Arxiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts) and place it in the project directory.


3. ### Run the application:
Once the dependencies are installed, run the Streamlit app by using the following command:

``streamlit run app.py``


