import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_test, y_pred):
    """Plot a confusion matrix
    
    Args:
        y_test (pandas series, nparray): Target test set
        y_pred (pandas series, nparray): Predicted targets

    Returns:
        Figure containing the confusion matrix
    """    

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


def plot_roc_auc(fpr, tpr, auc):
    """Plot a ROC curve
    
    Args:
        fpr (pandas series, nparray): false positive rate
        tpr (pandas series, nparray): true positive rate
        auc (float) : auc model score
        
    Returns:
        Figure containing the ROC curve and its AUC.
    """  
    
    plt.rcParams['figure.figsize'] = (12., 8.)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.legend(loc=4)
    