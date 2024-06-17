import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
import pickle

class LstmModel:
    """
    A class for mutilabel classification using an LSTM
    """
    def __init__(self,
                 word_index,
                 mlb,
                 max_seq_len:int=500
                 ):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(word_index)+1, output_dim=128, input_length=max_seq_len))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(len(mlb.classes_), activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self,
              padded_sequences,
              labels,
              model_name,
              epochs:int=5,
              batch_size:int=128,
              validation_split:float=0.2):
        
        history = self.model.fit(padded_sequences, 
                                 labels, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 validation_split=validation_split)

        with open(model_name, 'wb') as file:  
            pickle.dump(self.model, file)

        return history, self.model