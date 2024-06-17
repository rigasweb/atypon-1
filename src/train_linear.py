from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import ast 
from collections import Counter
import os 
from tqdm import tqdm
from utils.evaluate import Evaluator

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)


def create_dataset(data_path:str) -> None:
    """
    Prepare the dataset to train one linear classifier for each label
    For each label take the same number of positive and negative samples

    :param data_path: the path to the csv file containing the data
    :return: None
    """
    data = pd.read_csv(data_path)
    data["meshroot"] = data["meshroot"].apply(ast.literal_eval)  # apply literal_eval because lists appear as strings

    # get the unique values of the labels
    all_labels = [label for sublist in data['meshroot'] for label in sublist]
    label_counts = Counter(all_labels)
    labels = [label[0] for label in label_counts.items()]

    embedder = SentenceTransformer("neuml/pubmedbert-base-embeddings")
    X_all = []
    y_all = []
    n_samples = 10000

    for label in tqdm(labels):
        positive_samples = data[data["meshroot"].apply(lambda c: label in c)] # get all the rows that contain the label
        positive_samples = positive_samples[:min(n_samples,len(positive_samples))] 
        negative_samples = data[data["meshroot"].apply(lambda c: label not in c)]
        negative_samples = negative_samples.sample(n=len(positive_samples), replace=True) # sample the same number of rows that do not contain it

        X = []
        for text in positive_samples["abstractText"]:
            X.append(embedder.encode(text))
        for text in negative_samples["abstractText"]:
            X.append(embedder.encode(text))

        y = [label] * len(positive_samples) + ['0'] * len(negative_samples) # we use "0" as a negative label

        X_all.append(X)
        y_all.append(y)
    
    # Save the lists to a file
    with open('X.pkl', 'wb') as f:
        pickle.dump(X_all, f)
    with open('y.pkl', 'wb') as f:
        pickle.dump(y_all, f)


def train_model(X_train:list, y_train:list) -> LogisticRegression:
  """
  Train a linear classifier for a given label
  and save the model as a pickle file

  :param X: the embeddings for the training set
  :param y: the labels for the training set
  :param label: the name of the label
  :return: None
  """
  model = LogisticRegression()
  model.fit(X_train, y_train)

  """with open(f'classifiers/linear/{label}_classifier.pkl', 'wb') as file:
    pickle.dump(model, file)
  print(f'{label} classifier saved!')
  """
  return model  


if __name__ == "__main__":

    # create the dataset
    #create_dataset(f'{parent_directory}/data/articles.csv')

    # load the training data
    with open(f'{parent_directory}/data/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(f'{parent_directory}/data/y.pkl', 'rb') as f:
        y = pickle.load(f)

    # read the labels
    labels_df = pd.read_csv(f"{parent_directory}/data/labels.csv")
    labels = labels_df["Labels"].to_list()

    # for each label train a linear classifier
    evaluator = Evaluator()
    y_pred_all = []
    y_test_all = []
    pos_labels_all = []
    for idx, X_label in enumerate(X):  
        X_train, X_test, y_train, y_test = train_test_split(X_label,y[idx], test_size=0.2, random_state=42)

        print("start training for",labels[idx])
        model = train_model(X_train,y_train)

        y_pred, pos_label = evaluator.evaluate_linear(model, X_test, y_test)
        y_pred_all.append(y_pred)
        y_test_all.append(y_test)
        pos_labels_all.append(pos_label)

"""    report = classification_report(y_test_all, y_pred_all, target_names=labels)
    print("Classification Report:")
    print(report)
    

    # Calculate Hamming Loss
    hl = hamming_loss(y_test_all, y_pred_all)
    print(f"Hamming Loss: {hl}")"""
