import numpy as np
import json
import nltk
from nltk.stem import PorterStemmer
import random
import joblib
import streamlit as st
import time

ps = PorterStemmer()

def preprocess_text(text):

    words = nltk.word_tokenize(text)
#     words = [ps.stem(word.lower()) for word in words if word.isalnum()]  # Apply stemming
    return words

def create_bow(sentence, vocab):
#     sentence_words = preprocess_text(sentence)
    words = nltk.word_tokenize(sentence)
    words = [ps.stem(w.lower()) for w in words if w.isalnum()]
    sentence_words = sorted(list(set(words)))
    bow = [1 if word in sentence_words else 0 for word in vocab]
    return np.array(bow)

model = joblib.load('model2.h5') 

def extract_car_name(user_input):
    # Tokenize and lower the input
    tokens = user_input.lower().split()
    
    # Check for matching tokens in known car names
    for car in car_details:
        car_tokens = car.split()
        if all(token in tokens for token in car_tokens):
            return car
    return None

def get_car_details(car_name):
    car_name = car_name.lower()
    if car_name in car_details:
        return car_details[car_name]
    return None

with open("/Users/apple/Desktop/testing/dataset.json") as file:
    car_details = json.load(file)

with open("/Users/apple/Desktop/testing/intents.json") as file:
    data = json.load(file)

words =[]
for intents in data["intents"]:
    for patterns in intents["patterns"]:
        words.extend(preprocess_text(patterns))

words = [ps.stem(w.lower()) for w in words if w.isalnum()]
words = sorted(list(set(words)))
words = np.asarray(words)

classes = ['greeting','goodbye','thanks','show_cars','buy_car','car_price','car_details','car_speed','garbage']

def preprocess_input(user_input):
    # Tokenize and stem the user input
    user_input = nltk.word_tokenize(user_input)
    user_input = [ps.stem(word.lower()) for word in user_input]
    
    # Create a bag-of-words vector
    bag = [0] * len(words)
    for w in user_input:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array([bag])

def classify_intent(user_input):
    # Preprocess the input and predict the intent
    bow = preprocess_input(user_input)
    res = model.predict(bow)[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(intents_list, intents_json,user_input):
    tag = intents_list[0][0]
    intent_tag = classes[tag]
    
    # Find the corresponding intent and select a random response
    if intent_tag == "car_price":
        car = extract_car_name(user_input)
        return random.choice(data['intents'][tag]['responses'])+car_details[car]['price']
    elif intent_tag == "buy_car":
        car = extract_car_name(user_input)
        return random.choice(data['intents'][tag]['responses'])+car_details[car]['location']
    elif intent_tag == "car_details":
        car = extract_car_name(user_input)
        return random.choice(data['intents'][tag]['responses'])+car_details[car]['details']
    elif intent_tag == "car_speed":
        car = extract_car_name(user_input)
        return random.choice(data['intents'][tag]['responses'])+car_details[car]['speed']
    elif intent_tag == "show_cars":
        detail = ""
        for car in car_details:
            detail+=car+"\n"
        return detail
    
        
        
    for intent in intents_json['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])

def chatbot_response(user_input):
    intents_list = classify_intent(user_input)
    response = get_response(intents_list, data,user_input)
    return response


image="/Users/apple/Desktop/car.jpg"
# Inject custom CSS to set the background image
st.image(image, caption=None, width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("WELCOME TO PININFARINA")

for message in st.session_state.chat_history:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant"):
            st.write(message['content'])

        
# Accept user input
if prompt := st.chat_input("What is up?"):
    
    # Add user message to chat history
    #prediction = model.predict(np.array([bag_of_words(prompt, words, stemmer)]))
    #prediction = prediction[0]
    #results_index = np.argmax(prediction)
    #tag = labels[results_index]

    response = chatbot_response(prompt)

    # Get the response
    #response = get_response(tag,prompt)
    
    p=prompt
    r=response

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    placeholder = st.empty()
    message = ""
    
    # Add each word to the message string and update the placeholder
    for word in response.split(" "):
        message += word + " "  # Add word to message
        placeholder.text(message)  # Update the placeholder text
        time.sleep(0.08)  # Delay between each word

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
