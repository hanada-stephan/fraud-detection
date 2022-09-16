from sklearn.metrics import accuracy_score,\
                            f1_score,\
                            precision_score,\
                            recall_score


def print_scores(y_test, y_pred):
    """Print accuracy, precision, recall and f1 scores for classification models
    
    Args:
        y_test (Pandas series, nparray): Target of test data set
        y_pred (Pandas series, nparray): Target predicted

    Returns:
        Four prints for the model scores
    """

    print("Acurácia:",accuracy_score(y_test, y_pred))
    print("Precisão:",precision_score(y_test, y_pred))
    print("Recall:",recall_score(y_test, y_pred)) 
    print("F1:",f1_score(y_test, y_pred))
