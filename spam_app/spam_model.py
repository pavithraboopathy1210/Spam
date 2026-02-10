import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
df.columns = ['label', 'message']

# Convert labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Text to numbers
cv = CountVectorizer()
X = cv.fit_transform(df['message'])
y = df['label_num']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully")
