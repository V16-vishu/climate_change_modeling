import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Load
DATA_PATH = Path("data/raw/climate_data.csv")
df = pd.read_csv(DATA_PATH)

# Clean
from  prepare_data import normalize_text
df['text_clean'] = df['Text'].astype(str).map(normalize_text)

# Sentiment
analyzer = SentimentIntensityAnalyzer()
df['compound'] = df['text_clean'].map(lambda t: analyzer.polarity_scores(t)['compound'])
df['sent_label'] = pd.cut(df['compound'], bins=[-1, -0.05, 0.05, 1], labels=['neg','neu','pos'])
print("\nSample with sentiment:")
print(df[['Text','compound','sent_label']].head())

print("\nSentiment distribution:")
print(df['sent_label'].value_counts())

# Topics (unsupervised)
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(df['text_clean'])

nmf = NMF(n_components=5, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_

terms = vectorizer.get_feature_names_out()
for i, topic in enumerate(H):
    top = [terms[j] for j in topic.argsort()[::-1][:10]]
    print(f"\nTopic {i}: {', '.join(top)}")

# Save results
df.to_csv("../data/comments_with_sentiment_topics.csv", index=False)
print("\nSaved file with sentiment + topics → data/comments_with_sentiment_topics.csv")