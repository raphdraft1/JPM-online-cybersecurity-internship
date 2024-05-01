import pandas as pd
import os

def read_spam():
    category = 'spam'
    directory = './enron1/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = './enron1/ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

ham_emails = read_ham()
spam_emails = read_spam()

ham_df = pd.DataFrame.from_records(ham_emails)
spam_df = pd.DataFrame.from_records(spam_emails)

print("Number of Ham Emails:", len(ham_df))
print("Number of Spam Emails:", len(spam_df))


#2 Defining preprocessor
import re

def preprocessor(e):
    processed_text = re.sub(r'[^a-zA-Z]', ' ', e)
    processed_text = processed_text.lower()
    return processed_text

#3 ML model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Instantiate CountVectorizer
count_vectorizer = CountVectorizer(preprocessor=preprocessor)

# Split the dataset into train and test sets
X_train_ham, X_test_ham, y_train_ham, y_test_ham = train_test_split(ham_df["content"], ham_df["category"], test_size=0.2, random_state=1)
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(spam_df["content"], spam_df["category"], test_size=0.2, random_state=1)

# Concatenate the training and testing sets for both ham and spam
X_train = pd.concat([X_train_ham, X_train_spam])
X_test = pd.concat([X_test_ham, X_test_spam])
y_train = pd.concat([y_train_ham, y_train_spam])
y_test = pd.concat([y_test_ham, y_test_spam])

# Transform the dataset using CountVectorizer
X_train_transformed = count_vectorizer.fit_transform(X_train)
X_test_transformed = count_vectorizer.transform(X_test)

# Fit Logistic Regression model to the train dataset
logreg = LogisticRegression()
logreg.fit(X_train_transformed, y_train)

# Generate predictions on the test dataset
y_pred = logreg.predict(X_test_transformed)

# Evaluate the model
print(f'Accuracy:\n{accuracy_score(y_test, y_pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
print(f'Detailed Statistics:\n{classification_report(y_test, y_pred)}\n')

#4 categorize
# Let's see which features (aka columns) the vectorizer created. 
# They should be all the words that were contained in the training dataset.
# Get the vocabulary (terms) from the CountVectorizer object
vocabulary = count_vectorizer.vocabulary_

# Get the feature names (words) from the vocabulary
features = list(vocabulary.keys())

# You may be wondering what a machine learning model is tangibly. It is just a collection of numbers. 
# You can access these numbers known as "coefficients" from the coef_ property of the model
# We will be looking at coef_[0] which represents the importance of each feature.
# What does importance mean in this context?
# Some words are more important than others for the model.
# It's nothing personal, just that spam emails tend to contain some words more frequently.
# This indicates to the model that having that word would make a new email more likely to be spam.
importance = logreg.coef_[0]

# Iterate over importance and find the top 10 positive features with the largest magnitude.
# Similarly, find the top 10 negative features with the largest magnitude.
# Positive features correspond to spam. Negative features correspond to ham.
# You will see that `http` is the strongest feature that corresponds to spam emails. 
# It makes sense. Spam emails often want you to click on a link.
# Create a list of tuples containing index and importance
indexed_importance = list(enumerate(importance))

# Sort the list based on importance (descending order)
indexed_importance.sort(key=lambda e: e[1], reverse=True)

# Print the top 10 features with highest importance
print("Top 10 positive features (associated with spam):")
for i, imp in indexed_importance[:10]:
    print(imp, feature_names[i])

# Sort the list in reverse order to get top negative features
indexed_importance.sort(key=lambda e: -e[1], reverse=True)

# Print the top 10 features with lowest importance (negative)
print("\nTop 10 negative features (associated with ham):")
for i, imp in indexed_importance[:10]:
    print(imp, feature_names[i])