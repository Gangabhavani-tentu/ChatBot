from flask import Flask, render_template, request,jsonify
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
import json 
import random
app = Flask(__name__)
app.static_folder = 'static'
model = joblib.load(r'D:\Miniproject\chat_bot\chat_bot\model.pkl')
intents = json.loads(open('intents.json').read())
text=joblib.load('text.pkl')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text)
# y = labels
def generate_response(user_input):
    # user_input = request.args.get('msg')
    print('\n\n\n\n',user_input)
    input_text = vectorizer.transform([user_input])
    predicted_intent = model.predict(input_text)[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break       
    return response
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get_response",methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data['message']
    response = generate_response(user_message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run()

