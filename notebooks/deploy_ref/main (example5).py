import uvicorn
from fastapi import FastAPI
from model import SentimentModel, SentimentQueryModel

app = FastAPI()
model = SentimentModel()

@app.post('/predict')
def predict(data: SentimentQueryModel):
    data = data.dict()

    polarity, subjectivity = model.get_sentiment(data['text'])

    return {'polarity': polarity, 'subjectivity': subjectivity, 'comment': 'Example 5: deployed using FastAPI and Docker.'}

if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
