# Home Services Application

## Introduction
Welcome to the Home Services Application! This application leverages natural language processing and machine learning models to provide you with sentiment analysis, service retrieval, question answering, and review summarization. Below, we'll provide an overview of each functionality and how to use them.

## Sentiment Prediction
Our application uses the BERT multilingual cased sentiment model from Hugging Face to predict the sentiment of a given text. Simply enter your text in the provided text area and click the "Analyze Sentiment" button. The application will display the overall sentiment along with detailed scores.

## Service Retrieval
Our service retrieval functionality utilizes TF-idf vectorization and cosine similarity to match user queries with available services. The results are then weighted by the average score of each service to provide the best services. Enter your request in the provided text area and click the "Retrieve Services" button.

## Question Answering Chatbot
Our chatbot combines TF-idf and a question-answering model from Hugging Face to provide answers to user queries. Enter your question in the chat input, and the chatbot will respond based on relevant information retrieved using TF-idf.

## Review Summary
The review summary functionality generates a summary of a given review. Enter your review in the provided text area and click the "Generate Summary" button. Note that you may need to replace the summarization logic in the code snippet with your own implementation.

## Repository Content
- **`app_streamlit_nlp_project2.py`**: Python script to run the application.
- **`notebook_project2.ipynb`**: Jupyter notebook containing the project work.
- **`dataset`**:  containing the dataset used in the project.
- **`bestmodel.pkl`**: the model in pkl format to predict the rating 
- **`Trustpilot.ipynb`**: Jupyter notebook containing the code to scrap from Trustpilot, as it was too heavy to run in Google Colab.
- **`Yelp.ipynb`**: Jupyter notebook containing the code to scrap from Yelp, as it was too heavy to run in Google Colab.
- **`T5_Summary.ipynb`**: Jupyter notebook containing the abstrctive summary model, and its output, as it was too heavy to run in Google Colab.
- **`word2vec_imdb.model`**: Pre-trained Word2Vec model for the ML Sentiment analysis part.
## Streamlit App
You an access the application with this link : https://nlp-services.streamlit.app/
Link to the colab notebook : https://colab.research.google.com/drive/1vFqinydgpA5K11Ld4YfAq0Bzn1XPlVEU?usp=sharing
