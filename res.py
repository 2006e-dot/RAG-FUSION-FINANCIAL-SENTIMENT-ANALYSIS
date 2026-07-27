import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

FILE1="predictions_train.csv"
FILE2="predictions_valid.csv"

df1=pd.read_csv(FILE1)
df2=pd.read_csv(FILE2)

df2["doc_id"]=df2["doc_id"]+df1["doc_id"].max()+1

df=pd.concat([df1,df2],ignore_index=True)

y_true=df["true_label"].str.lower()
y_pred=df["predicted_label"].str.lower()

accuracy=accuracy_score(y_true,y_pred)

macro_precision,macro_recall,macro_f1,_=precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro"
)

weighted_precision,weighted_recall,weighted_f1,_=precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted"
)

print("="*60)
print(f"Samples            : {len(df)}")
print(f"Accuracy           : {accuracy*100:.2f}%")
print(f"Macro Precision    : {macro_precision*100:.2f}%")
print(f"Macro Recall       : {macro_recall*100:.2f}%")
print(f"Macro F1           : {macro_f1*100:.2f}%")
print(f"Weighted Precision : {weighted_precision*100:.2f}%")
print(f"Weighted Recall    : {weighted_recall*100:.2f}%")
print(f"Weighted F1        : {weighted_f1*100:.2f}%")

print("\nClassification Report\n")

print(classification_report(
    y_true,
    y_pred,
    digits=4
))

print("\nConfusion Matrix\n")

labels=["positive","neutral","negative"]

cm=confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df=pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)