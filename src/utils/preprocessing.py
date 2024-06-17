import re
import nltk
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import pickle

class Preprocessor:
    """
    A class for preprocessing and tokenizing the data
    """
    def __init__(self
                 ):
        nltk.download('stopwords')

    def clean_text(self,
                   text: str
                   ) -> str:
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\W', ' ', text)
        text = text.lower()
        text = text.strip()
        return text
    
    def clean_data(self,
                   data:pd.DataFrame
                   ) -> pd.DataFrame:
        data["cleaned_text"] = data['abstractText'].apply(self.clean_text)
        
        # Encoding labels
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(data['meshroot'])

        return data, labels, mlb
    
    def tokenize(self,
                 texts:list
                 ):

        # Tokenization
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['cleaned_text'])
        
        with open('tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)"""
        
        # Load the tokenizer
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index

        # Padding sequences
        max_seq_len = 500
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_len)

        return word_index, padded_sequences
    

class PubMedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)