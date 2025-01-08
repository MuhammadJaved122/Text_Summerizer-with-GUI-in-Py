import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from transformers import pipeline
import streamlit as st

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

# Preprocess the text
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

# Extractive summarization using TF-IDF
def extractive_summary_tfidf(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    preprocessed_sentences = [" ".join(preprocess_text(sentence)) for sentence in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).flatten().tolist()[0]

    ranked_sentences = sorted(
        [(score, sentence) for score, sentence in zip(sentence_scores, sentences)],
        key=lambda x: x[0],
        reverse=True,
    )

    top_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]
    return " ".join(top_sentences)

# Abstractive summarization
def abstractive_summarization(text, model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name, device=-1)  # Use CPU (-1)
    result = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return result[0]["summary_text"]

# Combined summarization pipeline
def combined_summarization_pipeline(text, num_sentences=3):
    # Extractive summarization
    extractive_summary = extractive_summary_tfidf(text, num_sentences=num_sentences)

    # Abstractive summarization
    abstractive_summary = abstractive_summarization(extractive_summary)

    return extractive_summary, abstractive_summary

# Streamlit App
def main():
    st.title("Text Summarization App")
    st.write("Combine extractive and abstractive summarization techniques to summarize your text.")

    # User input
    text_input = st.text_area("Enter the text to summarize:", height=200)
    num_sentences = st.slider("Number of sentences for extractive summarization:", 1, 10, 3)

    if st.button("Summarize"):
        if not text_input.strip():
            st.error("Please enter some text to summarize!")
        else:
            with st.spinner("Summarizing..."):
                extractive_summary, abstractive_summary = combined_summarization_pipeline(text_input, num_sentences=num_sentences)

            # Display results
            st.subheader("Extractive Summary")
            st.write(extractive_summary)

            st.subheader("Abstractive Summary")
            st.write(abstractive_summary)

if __name__ == "__main__":
    main()
