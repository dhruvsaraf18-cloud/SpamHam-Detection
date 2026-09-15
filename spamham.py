import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Spam_SMS.csv")
print(df["Class"].value_counts())
counts = df["Class"].value_counts()
counts.plot(kind="bar")
plt.title("Spam vs Ham SMS")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

df["Class"] = df["Class"].map({"ham": 0, "spam": 1})

x = df["Message"]
y = df["Class"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vectorized, y_train)

predictions = model.predict(x_test_vectorized)
message = input("Enter a message to classify as spam or ham: ")
message_vectorized = vectorizer.transform([message])
prediction = model.predict(message_vectorized)
if prediction[0] == 1:
    print("The message is classified as: SPAM")
else:
    print("The message is classified as: HAM")