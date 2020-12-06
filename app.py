#  things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import flask
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import json
import random
import traceback
# nltk.data.path.append('./nltk_data/')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


#Use pickle to load in the pre-trained model
model = pickle.load(open('model1.pkl', 'rb'))
data = pickle.load(open('data1.pkl', 'rb'))
words = data['words']
classes = data['classes']
print('Model data loaded')
with open('intents.json') as json_data:
    intents = json.load(json_data)

app = Flask(__name__, template_folder  =   "templates")


app.config["DEBUG"] = True
CORS(app)

@app.route("/", methods = ['GET'])
def home():
    return(flask.render_template("main.html"))


@app.route('/classify1', methods = ["POST"])
def classify1():
    ERROR_THRESHOLD = 0.9
    if model:
        try:
            sentence = flask.request.form['user']
            req = request.get_json()
            input_data = pd.DataFrame([bow(sentence, words)], dtype=float,
                                      )
            results = model.predict([input_data])[0]

            # filter out predictions below a threshold
            results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
            if len(results) == 0:
                final_response = "I dont seem to understand what you entered, can you please re-enter your input"
            else:
                # sort by strength of probability
                results.sort(key=lambda x: x[1], reverse=True)
                return_list = []
                for r in results:
                    intent_new = classes[r[0]]
                    return_list.append([classes[r[0]], str(r[1])])
                    i = 0
                    while i <= len(intents["intents"]):
                        check_intent = intents["intents"][i]["tag"]
                        check_responses = intents["intents"][i]["responses"]

                        if check_intent == intent_new:
                            final_response = random.choice(check_responses)
                            break
                        i += 1
            response = final_response

            return flask.render_template('main.html', Doctors_response="{}".format(response))
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ("No model here to use")



@app.route('/classify2', methods = ["POST"])
def classify2():
    ERROR_THRESHOLD = 0.9
    if model:
        try:
            sentence = request.json['sentence']
            req = request.get_json()
            input_data = pd.DataFrame([bow(sentence, words)], dtype=float,
                                      )
            results = model.predict([input_data])[0]

            # filter out predictions below a threshold
            results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
            if len(results) == 0:
                final_response = {
                    "response":"I dont seem to understand what you entered, can you please re-enter your input"
                }
            else:
                # sort by strength of probability
                results.sort(key=lambda x: x[1], reverse=True)
                return_list = []
                for r in results:
                    intent_new = classes[r[0]]
                    return_list.append([classes[r[0]], str(r[1])])
                    i = 0
                    while i <= len(intents["intents"]):
                        check_intent = intents["intents"][i]["tag"]
                        check_responses = intents["intents"][i]["responses"]

                        if check_intent == intent_new:
                            final_response = {
                                "response": random.choice(check_responses)
                            }
                            break
                        i += 1

            response = jsonify(final_response)

            return response
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ("No model here to use")



if __name__ == "__main__":
    # model = pickle.load(open('model1.pkl', 'rb'))
    # data = pickle.load(open('data1.pkl', 'rb'))
    # words = data['words']
    # classes = data['classes']
    # print('Model data loaded')
    # with open('intents.json') as json_data:
    #     intents = json.load(json_data)
    app.run(debug=False, threaded=False, host='127.0.0.1', port=4001)
