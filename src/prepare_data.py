import pandas as pd
from pathlib import Path
import re, html, unicodedata
import nltk

# Download VADER (only first time)
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer


OUT = Path("/data/processed/comments_with_sentiment.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Add normalize_text here
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = "".join(ch for ch in unicodedata.normalize("NFKC", s)
                if unicodedata.category(ch)[0] != "C")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def label_from_compound(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

def main():
    df = pd.read_csv(RAW)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df["sent_compound"] = df["Text"].map(lambda t: sia.polarity_scores(str(t))["compound"])
    df["sent_label"] = df["sent_compound"].map(label_from_compound)

    df.to_csv(OUT, index=False)
    print(f"Cleaned file saved at {OUT.resolve()}")

if __name__ == "__main__":
    main()