import pandas as pd
from NaiveBayes import frequencies, train_NB, predict_NB

# load the data
df_real = pd.read_csv("True.csv")
df_real['outcome'] = 1
df_fake = pd.read_csv("Fake.csv")
df_fake['outcome'] = 0

# combine the data
df = pd.concat([df_real, df_fake], axis=0)

#shuffling the df
df = df.sample(frac=1).reset_index(drop=True)

# split into training and testing sets
train_index = int(0.8 * len(df))
train_df = df[:train_index]
test_df = df[train_index:]

# prepare training data
train_df_text = train_df['text']
train_df_outcomes = train_df['outcome']

# prepare testing data
test_df_text = test_df['text']
test_df_outcomes = test_df['outcome']

# calculate word frequencies
word_freqs = frequencies(train_df_text, train_df_outcomes)

# train Naive Bayes model
prior_probabilities, word_likelihoods = train_NB(word_freqs, train_df_outcomes)

# predict on the test set and calculate accuracy
predictions = [predict_NB(text, prior_probabilities, word_likelihoods) for text in test_df_text]



#calculating accuracy
num_correct = 0
for test_outcome, test_prediction in zip(test_df_outcomes, predictions):
    if test_outcome == test_prediction:
        num_correct += 1

accuracy = num_correct/len(test_df_outcomes)*100

print(f"Accuracy: {accuracy:.2f}")