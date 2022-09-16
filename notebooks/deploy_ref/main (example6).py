from model import SentimentQueryModel, SentimentModel
from flask import Flask, request

app = Flask('sentiment_model')
model = SentimentModel()

@app.route('/', methods=['GET', 'POST'])
def predict():
    data = request.get_json()

    polarity, subjectivity = model.get_sentiment(data['text'])

    return {'polarity': polarity, 'subjectivity': subjectivity, 'comment': 'Example 6: deployed using Flask and Docker.'}

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8005, debug=True)
