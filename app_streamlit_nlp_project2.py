import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoModelWithLMHead, AutoTokenizer

import pickle
import string

from sklearn.svm import SVC

import time

# Load Course dataset

df=pd.read_csv("df_mergedV4.csv",sep=',')
filtered_documents=df.copy()

@st.cache(allow_output_mutation=True)
def load_sentiment_analysis_pipeline():
    return pipeline(
        task="sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
        return_all_scores=True
)

@st.cache(allow_output_mutation=True)
def load_qa_pipeline():
    return pipeline('question-answering')





@st.cache_data(persist=True)
def get_tfidf_vectorizer(description_trad_clean):
    # Create a TF-IDF vectorizer with specific settings
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    return tfidf_vectorizer.fit(description_trad_clean)

# Retrieve the TF-IDF vectorizer, cached for efficiency
tfidf_vectorizer = get_tfidf_vectorizer(filtered_documents['description_trad_clean'])


@st.cache_data(persist=True)
def retrieve_top_documents(query_summary, k=10):
    # Transform the query summary into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query_summary])
    
    # Calculate cosine similarity between the query and all articles
    similarity_scores = linear_kernel(query_vector, tfidf_vectorizer.transform(filtered_documents['description_trad_clean']))
    
    # Add the impact of the average score to the similarity scores
    similarity_scores_with_impact = similarity_scores + filtered_documents['average_score'].values.reshape(1, -1) * 0.1  # Adjust the impact factor as needed
    
    # Sort document indices by the modified similarity score in descending order
    document_indices = similarity_scores_with_impact[0].argsort()[:-k-1:-1]
    
    # Retrieve the top-k documents based on their indices
    top_documents = filtered_documents.iloc[document_indices] 
    
    return similarity_scores_with_impact, top_documents

# Function to create context string for top documents
def create_context_string(top_documents):
    unique_names = top_documents["name"].unique()

    contexts = []
    for name in unique_names:
        subset_df = df[df["name"] == name]
        avg_score = subset_df["average_score"].mean()
        description = subset_df['description_trad_clean'].iloc[0]
        phone_number = subset_df["phone_number"].iloc[0]  # Assuming it's the same for each entry
        context_string = f"The Name of the company is: {name},  {name}'s Description is: {description} {name}'s Average Score is: {avg_score} and {name}'s Phone Number is: {phone_number}"
        contexts.append(context_string)
    
    return '\n'.join(contexts)


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Add any other preprocessing steps as needed

    return text

@st.cache_data(persist=True)
def load_t5_tokenizer():
    return AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

@st.cache_data(persist=True)
def load_t5_model():
    return AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")




model_filename = 'svm_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

@st.cache_data(persist=True)
def load_svm_model():
    model_filename = 'svm_model.pkl'
    with open(model_filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model

@st.cache_data(persist=True)
def load_tfidf_vectorizer():
    vectorizer_filename = 'tfidf_vectorizer.pkl'
    with open(vectorizer_filename, 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    return loaded_vectorizer

def summarize(text,tokenizer, model, max_length=150):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds[0]

# Streamlit App
def main():
    st.title("Home services Application")

     # Sidebar with options
    page_options = ["Home", "Prediction","Service Retrieval","Chatbot: Question Answering","Summary","Explanation"]
    page = st.sidebar.selectbox("Choose a page", page_options)

    

    if page == "Home":
        
        st.image("https://global.hitachi-solutions.com/wp-content/uploads/2022/01/Webinar-NLP-In-RCG.png", width=750, caption="Welcome to the NLP App")
        
        st.markdown(
        """
        # Welcome to our NLP application on home services.
        In this application, you will find different use case of NLP techniques on a scrapped data base.
        Use The sidebar to navigate to different pages.

        Have Fun !

        Janany & Mathilde
        """
    )
    
    elif page == "Prediction":
        st.header("Sentiment Prediction")
        st.write("Enter a comment below to predict its sentiment:")

        # User input for prediction
        user_input = st.text_area("Enter your text here:")
        classification=load_sentiment_analysis_pipeline()

        if st.button("Analyze Sentiment"):
            # Perform sentiment analysis
            result = classification(user_input)
            # Determine the overall sentiment
            overall_sentiment = max(result[0], key=lambda x: x['score'])
            if overall_sentiment['label']=='negative':
                st.write("Sorry to hear that your experience with this service was not up to your expectations. We appreciate your feedback.")
            elif overall_sentiment['label']=='positif':
                st.write("We are glad of you experience with this service. We appreciate your feedback.")
            else :
                st.write("We appreciate your feedback.")
            st.write(f"Overall Sentiment: {overall_sentiment['label']}")

            with st.expander("Show Detailed Scores"):
                st.write("Sentiment Scores:")
                for score in result[0]:
                    st.write(f"{score['label']}: {score['score']}")
        # Add rating prediction below sentiment analysis
        st.header("Prediction")
        st.write("Enter a comment below to predict its rating:")
        # User input for rating prediction
        user_input_rating = st.text_area("Enter your review here:")

        if st.button("Predict Rating"):
             # Preprocess the user input text
            user_input_rating_processed = preprocess_text(user_input_rating)
            loaded_vectorizer=load_tfidf_vectorizer()
            loaded_model=load_svm_model()
            # Vectorize the input text using the loaded TF-IDF vectorizer
            user_input_vectorized = loaded_vectorizer.transform([user_input_rating_processed])

            # Predict the rating using the loaded SVM model
            rating_prediction = loaded_model.predict([user_input_rating_processed])[0]

            st.write(f"Predicted Rating: {rating_prediction}")




    elif page == "Service Retrieval":
        st.header("Service Retrieval Page")
        query_summary = st.text_area("✏️ Enter your request : Exemple: Small hand jobs in Paris ")

        if st.button("Retrieve Services"):
            if query_summary:
                # Retrieve top documents using TF-IDF
                similarity_scores_with_impact, top_documents = retrieve_top_documents(query_summary, k=10)

                # Display the query summary
                st.subheader("Your are looking for:")
                st.write(query_summary)

                # Display the top 10 results
                st.subheader("Top 10 Services:")
                for i, (index, row) in enumerate(top_documents.iterrows(), 1):
                    st.write(f"{i}. **{row['name']}**")
                    st.write(f"   Average Score: {row['average_score']:.2f} ⭐️")

                    # Calculate impact based on average score (customize the impact calculation as needed)
                    impact = row['average_score'] * 0.1  # Adjust the multiplication factor as needed
                   

                    # Display the modified similarity score with impact
                    st.write(f"   Similarity Score with Impact: {similarity_scores_with_impact[0][index]:.4f} 🚀")

                    # Expander for additional details
                    with st.expander("Show Details"):
                        st.write(f"   🎯 Description: {row['description_trad_clean']}")
                        st.write(f"   🔗 Link: {row['link']}")
                        st.write(f"   📍 Location: {row['location']}")
                        st.write(f"   📧 Email: {row['email']}")
                        st.write(f"   📞 Phone Number: {row['phone_number']}")

                    st.write("   ---")
            else:
                st.warning("Please enter a query summary before retrieving services.")
    

    elif page == "Chatbot: Question Answering":
        st.header("Try our Assistant chatbot to help you !")
        st.subheader("Hello! How can I help you? Ask for example: Can you give me the name of a company that do small home jobs in Paris ")

        
        # Display chat messages from history
        

        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "context_string" not in st.session_state:
            st.session_state.context_string=[]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            

        if prompt  and st.session_state.context_string==[] :
                # Retrieve top documents using TF-IDF
            similarity_scores_with_impact, top_documents = retrieve_top_documents(prompt, k=10)

               

                # Create context string for question answering
            context_string = create_context_string(top_documents)
            st.session_state.context_string = context_string

                # Perform question answering
            qa =load_qa_pipeline()
            answer = qa(context=context_string, question=prompt)

            with st.chat_message("assistant"):
                message_placeholder=st.empty()
                full_response=""
                assistant_response=answer['answer']

                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
             # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})



            
                # Option to refresh or continue with the same context
        elif prompt and st.session_state.context_string!=[]:
            qa =load_qa_pipeline()
            answer = qa(context=st.session_state.context_string, question=prompt)
            print("prompt",prompt)

            with st.chat_message("assistant"):
                message_placeholder=st.empty()
                full_response=""
                assistant_response=answer['answer']

                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
             # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
                st.text("Continue with the same context. Ask another question.")
                st.session_state.context_string=[]

    elif page=="Summary":
        st.header("Review Summarization Page")
        st.write("Enter your review below, and we will provide a summary for you.")
        #tokenizer = load_t5_tokenizer()
        #model = load_t5_model()

        # User input for review
        user_review = st.text_area("Enter your review here:")

        if st.button("Generate Summary"):
            # Perform summarization (you may need to replace this with your summarization logic)
            # For this example, let's assume a simple summary by taking the first 50 characters of the review
            #summary = summarize(user_review,tokenizer, model)
            summary="the model used for this part consume too much memory, you can try it on the colab https://colab.research.google.com/drive/1vFqinydgpA5K11Ld4YfAq0Bzn1XPlVEU?usp=sharing"
            
            st.subheader("Review Summary:")
            st.write(summary)

    elif page == "Explanation":
        st.header("Explanation Page")
        st.write("Welcome to the Explanation Page. Here, we'll provide some details about the app and its functionality.")

        # Add explanation text
        st.subheader("How the Sentiment Prediction Works:")
        st.write("Sentiment prediction is based on the bert multilingual cased sentiment available on Hugging Face. The model compute as result a dictionnary with the sentiment label as the key and its probability as value. In our application, we select the overall sentiment.")
        
        # Add code snippets
        st.subheader("Code Snippet: Sentiment Prediction")
        st.code("""
        classification = pipeline(
                                    task="sentiment-analysis",
                                    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
                                    return_all_scores=True)

        user_input = st.text_area("Enter your text here:")
        
        if st.button("Analyze Sentiment"):
            result = classification(user_input)
            # Process the sentiment analysis results
            # Display the overall sentiment and detailed scores
            """)

        st.subheader("How the Service Retrieval Works:")
        st.write("This functionnality is based on Tf-idf vectorization of each description of services. We then compute the cosine similiarity between the user's query and the available services. Finally, this score is weighted by the average score of each services, in order to provide the best services.")
        
        # Add code snippets
        st.subheader("Code Snippet: Service Retrieval")
        st.code("""
        query_summary = st.text_area("✏️ Enter your request :")
        
        if st.button("Retrieve Services"):
            similarity_scores_with_impact, top_documents = retrieve_top_documents(query_summary, k=10)
            # Process and display the top service retrieval results
            """)

        st.subheader("How the Question Answering Chatbot Works:")
        st.write("our Chatbot relieves on the combination of TF-idf and question answering model from Hugging Face.")
        
        # Add code snippets
        st.subheader("Code Snippet: Question Answering Chatbot")
        st.code("""
        qa = pipeline('question-answering')
        prompt = st.chat_input("What is up?")
        
        if prompt and st.session_state.context_string == []:
            similarity_scores_with_impact, top_documents = retrieve_top_documents(prompt, k=10)
                # we compute top 10 results that matches the user's query using tf-idf
            context_string = create_context_string(top_documents)
                #we store those info into a context string
            st.session_state.context_string = context_string
            answer = qa(context=context_string, question=prompt)
                # using the qa model, we provide the context and user's query
            # Display the chatbot response
            """)
        
        st.subheader("How the Summary works:")
        st.write("The summary is based on")
        
        # Add code snippets
        st.subheader("Code Snippet: Review summary")
        st.code("""
        
         user_review = st.text_area("Enter your review here:")

        if st.button("Generate Summary"):
            # Perform summarization (you may need to replace this with your summarization logic)
            # For this example, let's assume a simple summary by taking the first 50 characters of the review
            summary = summarize(user_review)
            
            st.subheader("Review Summary:")
            st.write(summary)
            """)

    





            

    

        




if __name__ == "__main__":
    main()
