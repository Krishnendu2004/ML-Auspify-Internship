import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("Dataset.csv")
print(data.head())

#STEP 1 : Feature Selection
data["director_known"] = np.where(data["director"] == "Not Given", 0, 1) # NG=0 else 1

inputs = data[["release_year", "director_known", "rating"]]
answer = np.where(data["type"] == "TV Show", 1, 0) #1=TV Show, 0=Movie

#STEP 2 : Label Encoding
inputs["rating"] = LabelEncoder().fit_transform(inputs["rating"])
print(inputs.head()) #Replaces Rating with numerical values -->  LabelEncoder().fit_transform(inputs["rating"])

#STEP 3 : Model Training
x_train, x_test, y_train, y_test = train_test_split(inputs, answer, test_size=0.2, random_state=42)

logistic = LogisticRegression(max_iter=2000)
logistic.fit(x_train, y_train)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(x_train, y_train)

forest = RandomForestClassifier(random_state=42)
forest.fit(x_train, y_train)

#STEP 4 : Prediction Evaluation
models = [logistic, tree, forest]
name = ["Logistic Regression", "Decision Tree", "Random Forest"]
accur = []

for i in range(3):
    prediction = models[i].predict(x_test)                  
    acc = accuracy_score(y_test, prediction)           
    accur.append(acc)

    print("\n" + name[i])
    print("Accuracy:", round(acc, 3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, prediction))
baseline = 1 - y_test.mean()
print("\nBaseline accuracy (always guess Movie):", round(baseline, 3))

#STEP 5 : Model Accuracy
bars = plt.bar(name + ["Baseline"], accur + [baseline], color=["red", "green", "orange", "yellow"])
plt.bar_label(bars, fmt="%.3f")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("Model_Evaluation.png")
plt.show()