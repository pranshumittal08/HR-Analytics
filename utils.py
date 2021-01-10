from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 


#Function to plot the precision recall curve
def plot_precision_recall_curve(y_true, pred_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, pred_prob)
    precision = list(precision)
    precision.pop()
    recall = list(recall)
    recall.pop()
    plt.plot(thresholds, precision, label = "Precision")
    plt.plot(thresholds, recall, label = "Recall")
    plt.xlabel("Threshold", fontsize= 14)
    plt.legend()
    plt.title("Precision Recall Curve", fontsize = 16)
    plt.show()


#Plot the f1_score curve for different threshold values
def plot_f1_curve(y_true, y_pred_prob):
    thresholds = [i/100 for i in range(0, 100, 5)]
    f1_scores = []
    for value in thresholds:
        y_pred = [1 if i >= value else 0 for i in y_pred_prob]
        f1_scores.append(round(f1_score(y_true, y_pred), 2))
    plt.plot(thresholds, f1_scores)
    plt.show()


#Display the confusion matrix
def show_confusion(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    confusion = pd.DataFrame(matrix, index = [["Actual","Actual"], ["Not_promoted", "Promoted"]], columns = ["Not_promoted", "Promoted"])
    print(confusion)

#Display Scores
def print_all_score(y_true, y_pred, data_set_type):
    precision = round(precision_score(y_true, y_pred),2)
    recall = round(recall_score(y_true, y_pred), 2)
    f1 = round(f1_score(y_true, y_pred), 2)
    print(f"The precision score on the {data_set_type} set is {precision}")
    print(f"The recall score on the {data_set_type} set is {recall}")
    print(f"The F1 score on the {data_set_type} set is {f1}")
    return precision, recall, f1

#Method to append each models results
def add_results(results_df, model,f1_train, f1_test, recall_train, recall_test, precision_train, precision_test, cut_off_prob):
        df = pd.DataFrame([[model, f1_train, f1_test, recall_train, recall_test, 
                            precision_train, precision_test, cut_off_prob]], 
                          columns = ["Algorithm","f1_train", "f1_test", 
                                      "Recall_train", "Recall_test", 
                                      "Precision_train", "Precision_test", "Cut_off_prob"])
        return results_df.append(df, ignore_index = True)