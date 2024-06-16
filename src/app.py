from fastapi import FastAPI, Request
import uvicorn
import pickle
from utils.preprocessing import Preprocessor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import ast
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os 

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

app = FastAPI()
data = pd.read_csv(f'{parent_directory}/data/articles.csv')
data["meshroot"] = data["meshroot"].apply(ast.literal_eval)  # apply literal_eval because lists appear as strings
embedder = SentenceTransformer("neuml/pubmedbert-base-embeddings")
 
class TextRequest(BaseModel):
    text: str


@app.post("/linear_predict")
async def linear_predict(request: TextRequest):
    """
    run all the classifiers for a given text
    and predict the labels
    """
    received_text = request.text
    sentence_embedding = embedder.encode(received_text)
    predicted_labels = []

    for classifier in os.listdir(f'{parent_directory}/src/classifiers/linear'):
        with open(f'{parent_directory}/src/classifiers/linear/{classifier}', 'rb') as file:
            loaded_model = pickle.load(file)

        y_pred = loaded_model.predict([sentence_embedding])
        # if the prediction is positive, save the label
        if y_pred != '0': predicted_labels.append(y_pred[0]) 

    return {"predicted class": predicted_labels}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)