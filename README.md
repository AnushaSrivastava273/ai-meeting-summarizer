# 📝 AI Meeting Summarizer

An **AI-powered Streamlit application** that helps teams save time by automatically summarizing meeting transcripts, extracting actionable items, and analyzing overall sentiment.  

This project leverages **state-of-the-art NLP models** from HuggingFace and spaCy to provide production-ready functionality.

---
# 📝 AI Meeting Summarizer

An AI-powered Streamlit app that **summarizes meeting transcripts**, extracts **action items**, and provides a **dashboard view** of the conversation.  

🔗 **Live Demo:** [Click Here](https://anushasrivastava273-ai-meeting-summarizer-app-lindtn.streamlit.app/)
---
## ✨ Features
- 📄 **Transcript Upload**: Upload raw meeting transcripts in `.txt` format  
- 🤖 **AI Summarization**: Generates concise summaries using **BART-Large-CNN**  
- ✅ **Action Item Detection**: Extracts key tasks & responsibilities via **spaCy NLP**  
- 😀 **Sentiment Analysis**: Detects positive/negative sentiment with **DistilBERT (SST-2)**  
- 📊 **Interactive Dashboard**: Displays:
  - Total words in transcript  
  - Estimated reading time  
  - Count of detected action items  

---

## 🤖 Models Used

### 1. **Summarization — BART-Large-CNN**
- Model: `facebook/bart-large-cnn`  
- Type: Transformer (Encoder-Decoder, seq2seq)  
- Purpose: Summarizes long transcripts into short, human-readable text.  
- Why BART? Pretrained on large-scale text summarization datasets, it’s optimized for abstractive summaries rather than just extractive snippets.  

---

### 2. **Sentiment Analysis — DistilBERT (SST-2)**
- Model: `distilbert-base-uncased-finetuned-sst-2-english`  
- Type: Distilled version of BERT (smaller, faster, cheaper to run)  
- Purpose: Classifies sentences/sections into **Positive** or **Negative** tone.  
- Why DistilBERT? Lighter & efficient, but retains ~97% of BERT’s performance. Perfect for quick real-time sentiment scoring in meetings.  

---

### 3. **Action Item Extraction — spaCy (en_core_web_sm)**
- Model: `en_core_web_sm`  
- Type: Rule-based + statistical NLP model  
- Purpose: Detects named entities, verbs, and task-like sentences (e.g., *“Alice will prepare slides”*)  
- Why spaCy? Lightweight, fast, and well-suited for information extraction tasks.  

---
👩‍💻 Tech Stack

Python 3.10+

Streamlit
 - frontend & dashboard

HuggingFace Transformers
 - summarization & sentiment analysis

spaCy
 - NLP pipeline for action item extraction
---
Dashboard

Action Items Example
- Bob will handle data analysis
- Charlie to prepare presentation slides
- Review meeting planned on Friday

AI Summary Example
The team discussed completing the project report by Monday. 
Bob will handle the analysis, Charlie the presentation, and 
a review meeting is scheduled for Friday.
