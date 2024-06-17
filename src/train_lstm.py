import pandas as pd 
import ast
from models import LstmModel
from utils.preprocessing import Preprocessor
from utils.evaluate import Evaluator


# load the dataset
data = pd.read_csv('../data/sampled_data.csv')
data["meshroot"] = data["meshroot"].apply(ast.literal_eval)  # apply literal_eval because lists appear as strings

 
# preprocess the data 
preprocessor = Preprocessor()
data, labels, mlb = preprocessor.clean_data(data)
word_index, padded_sequences = preprocessor.tokenize(data["cleaned_text"])


# build the model
lstm = LstmModel(word_index, mlb)
history, lstm_trained = lstm.train(padded_sequences, labels, "src/classifiers/lstm_1.pkl")


# evaluate
evluator = Evaluator()
evluator.evaluate_lstm(lstm_trained, padded_sequences, labels, mlb)