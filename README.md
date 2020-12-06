# MedicalBotAPI
_This Repository contains the Natural Language Processing model created using Tensorflow used to generate responses for the medical chatbot I created based on some of the users input from the chatbot. Flask was used to create the backend that hosts the API used to connect the model to the chatbot. The app was finally hosted on heroku._

To train the model, run:
`python model.py`
The above generates the pickle file to be used in the flask app

The data used to train the model is in `intents.json`

To run the flask server on your local machine, run the code below:
`python app.py`
The above command will result in a web interface to test the chatbot model.


