import streamlit as st
from transformers import pipeline
import re
import spacy
import math
from collections import Counter

def extract_action_items(text):
    # Regex pattern to find sentences with action-oriented keywords
    action_keywords = r'\b(will|shall|must|need to|required to)\b'
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    action_items = [sentence for sentence in sentences if re.search(action_keywords, sentence, re.IGNORECASE)]

    # Load spaCy model for imperative sentence detection
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    imperative_sentences = [sent.text for sent in doc.sents if sent.root.tag_ == "VB"]

    # Combine regex and spaCy results
    combined_action_items = list(set(action_items + imperative_sentences))
    return combined_action_items

def chunk_text(text, max_chunk_size=1000):
    """Split text into smaller chunks for summarization."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def calculate_stats(transcript, action_items):
    """Calculate quick stats for the transcript."""
    total_words = len(transcript.split())
    reading_time = math.ceil(total_words / 200)  # Assuming 200 words per minute
    num_action_items = len(action_items)
    return total_words, reading_time, num_action_items

def analyze_sentiment(transcript):
    """Perform sentiment analysis on the transcript."""
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcript)
    sentiment_results = sentiment_analyzer(sentences)

    sentiment_counts = Counter([result['label'] for result in sentiment_results])
    positive_sentences = [sentences[i] for i, result in enumerate(sentiment_results) if result['label'] == 'POSITIVE']
    negative_sentences = [sentences[i] for i, result in enumerate(sentiment_results) if result['label'] == 'NEGATIVE']

    # Get top 3 most positive and negative sentences
    top_positive = positive_sentences[:3]
    top_negative = negative_sentences[:3]

    return sentiment_counts, top_positive, top_negative

def main():
    st.title("AI Meeting Summarizer")

    # File uploader
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        # Read the file content
        transcript = uploaded_file.read().decode("utf-8")

        # Extract action items
        action_items = extract_action_items(transcript)

        # Calculate stats
        total_words, reading_time, num_action_items = calculate_stats(transcript, action_items)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Words", total_words)
        col2.metric("Reading Time (mins)", reading_time)
        col3.metric("Action Items", num_action_items)

        # Display tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Raw Transcript", "AI Summary", "Action Items", "Sentiment Analysis"])

        # Tab 1: Raw Transcript
        with tab1:
            st.text_area("Transcript", transcript, height=300)

        # Summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunks = chunk_text(transcript)
        summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
        combined_summary = " ".join(summaries)

        # Tab 2: AI Summary
        with tab2:
            st.write(combined_summary)

        # Tab 3: Action Items
        with tab3:
            if action_items:
                st.markdown("\n".join([f"- {item}" for item in action_items]))
            else:
                st.write("No action items found.")

        # Tab 4: Sentiment Analysis
        with tab4:
            sentiment_counts, top_positive, top_negative = analyze_sentiment(transcript)

            # Display bar chart of sentiment counts
            st.bar_chart(sentiment_counts)

            # Display top positive sentences
            with st.expander("Top 3 Positive Sentences"):
                for sentence in top_positive:
                    st.write(sentence)

            # Display top negative sentences
            with st.expander("Top 3 Negative Sentences"):
                for sentence in top_negative:
                    st.write(sentence)

if __name__ == "__main__":
    main()
