# Day 5-7 NLP Preprocessing + Visualization

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# -------------------------------
# Step 0: Download NLTK resources (fix LookupError)
# -------------------------------
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------
# Step 1: Load CSV
# -------------------------------
data = pd.read_csv('test.csv')  # Make sure your CSV filename matches
print("✅ CSV loaded successfully!")
print(data.head())

# -------------------------------
# Step 2: NLP Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation

def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in punctuation])  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

data['cleaned_text'] = data['text'].apply(clean_text)
print("\n✅ Text cleaned successfully!")
print(data[['text', 'cleaned_text']].head())

# -------------------------------
# Step 3: Tokenization
# -------------------------------
data['tokens'] = data['cleaned_text'].apply(word_tokenize)
print("\n✅ Text tokenized successfully!")
print(data[['cleaned_text', 'tokens']].head())

# -------------------------------
# Step 4: Word Cloud
# -------------------------------
all_text = ' '.join(data['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud")
#plt.show()
wordcloud.to_file('wordcloud.png')
print("✅ Word cloud saved as wordcloud.png")

# -------------------------------
# Step 5: Top 10 Frequent Words
# -------------------------------
all_words = [word for tokens in data['tokens'] for word in tokens]
word_freq = Counter(all_words)

freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'count']).sort_values(by='count', ascending=False)
print("\nTop 10 frequent words:")
print(freq_df.head(10))

plt.figure(figsize=(10,5))
sns.barplot(x='count', y='word', data=freq_df.head(10))
plt.title('Top 10 Most Frequent Words')
plt.savefig('top_words.png')
#plt.show()
print("✅ Top words plot saved as top_words.png")

# -------------------------------
# Step 6: Save processed CSV
# -------------------------------
data.to_csv('processed_text.csv', index=False)
print("✅ Processed CSV saved as processed_text.csv")

# -------------------------------
# Step 6: Save processed CSV
# -------------------------------
import os

print("💾 Current working directory:", os.getcwd())
data.to_csv('processed_text.csv', index=False)
print("✅ Processed CSV saved as processed_text.csv")

