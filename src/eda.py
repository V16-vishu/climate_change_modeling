import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from prepare_data import normalize_text

DATA_PATH = Path("data/raw/climate_data.csv")

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))

# Clean
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['text_clean'] = df['text'].astype(str).map(normalize_text)
df['word_len'] = df['text_clean'].str.split().str.len()
df['commentsCount'] = pd.to_numeric(df['commentsCount'], errors='coerce')

# Missing values
print("\nMissing values:")
print(df.isna().sum())

# Distribution plots
df['likesCount'].plot(kind='hist', bins=30, title="likes Count Distribution")
plt.show()

df['commentsCount'].plot(kind='hist', bins=30, title="comments Count Distribution")
plt.show()

df['word_len'].plot(kind='hist', bins=30, title="Words per Comment")
plt.show()

# Time trend
by_month = df.dropna(subset=['date']).groupby(df['date'].dt.to_period('M')).size()
by_month.plot(title="Comments per Month")
plt.show()