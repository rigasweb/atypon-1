from sklearn.metrics import classification_report, hamming_loss
from sklearn.metrics import classification_report, hamming_loss, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

class Evaluator:
    """
    A class for evaluating multiple models
    """
    def __init__(self):
        return
    
    def evaluate_linear(self, model:LogisticRegression, X_test:list, y_test:list) -> list:
        """
        evaluate e given model and calculate accuracy, precision, recall and f1score

        :param model: the model to be evaluated
        :param X_test: the x testing set
        :param y_test: the y testing set
        :return: the predictions
        """
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        pos_label = [cls for cls in model.classes_ if cls != '0'][0]
        precision = precision_score(y_test, y_pred, average="binary", pos_label=pos_label)
        recall = recall_score(y_test, y_pred, average="binary", pos_label=pos_label)
        f1 = f1_score(y_test, y_pred, average="binary", pos_label=pos_label)
        print(f"{pos_label}: Accuracy:{accuracy} Precision:{precision}, Recall:{recall}, f1:{f1}")

        return y_pred, pos_label

    def evaluate_lstm(self, 
                      model,
                      padded_sequences,
                      labels,
                      mlb):
        # Predict on validation set
        predictions = model.predict(padded_sequences)

        # Convert predictions to binary format
        predictions_binary = (predictions > 0.5).astype(int)

        # Evaluate using classification report
        print(classification_report(labels, predictions_binary, target_names=mlb.classes_))

        # Evaluate using Hamming Loss
        print('Hamming Loss:', hamming_loss(labels, predictions_binary))