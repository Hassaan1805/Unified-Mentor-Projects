import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

playstore_df = pd.read_csv('googleplaystore.csv')
reviews_df = pd.read_csv('googleplaystore_user_reviews.csv')

playstore_df['Installs'] = playstore_df['Installs'].replace('[+,]', '', regex=True).replace('Free', '0').astype(int)

playstore_df['Price'] = playstore_df['Price'].replace('[\$,]', '', regex=True).replace('Everyone', '0').astype(float)

playstore_df['Reviews'] = pd.to_numeric(playstore_df['Reviews'], errors='coerce')

playstore_df.fillna(0, inplace=True)

playstore_df['popularity'] = playstore_df['Installs'] + playstore_df['Reviews']

avg_rating_per_category = playstore_df.groupby('Category')['Rating'].mean().reset_index().rename(columns={'Rating': 'avg_rating_per_category'})

paid_apps_df = playstore_df[playstore_df['Price'] > 0]
plt.figure(figsize=(10, 6))
sns.histplot(paid_apps_df['Price'], bins=50, kde=True)
plt.title('Price Distribution for Paid Apps')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

playstore_df['revenue'] = playstore_df['Price'] * playstore_df['Installs']

sentiment_group = reviews_df.groupby('App')['Sentiment'].value_counts().unstack().fillna(0)
sentiment_group['avg_sentiment'] = (sentiment_group['Positive'] - sentiment_group['Negative']) / (sentiment_group['Positive'] + sentiment_group['Negative'] + sentiment_group['Neutral'])

playstore_df['reviews_per_install'] = playstore_df['Reviews'] / playstore_df['Installs']
playstore_df['reviews_per_install'].fillna(0, inplace=True)

playstore_df['Last Updated'] = pd.to_datetime(playstore_df['Last Updated'], errors='coerce')
playstore_df['days_since_update'] = (pd.Timestamp.now() - playstore_df['Last Updated']).dt.days.fillna(0)

rating_by_content = playstore_df.groupby('Content Rating')['Rating'].mean().reset_index()

sentiment_count_per_app = reviews_df.groupby('App')['Sentiment'].value_counts().unstack().fillna(0)

playstore_df['normalized_rating'] = (playstore_df['Rating'] - playstore_df['Rating'].min()) / (playstore_df['Rating'].max() - playstore_df['Rating'].min())

playstore_df['log_installs'] = np.log1p(playstore_df['Installs'])

plt.figure(figsize=(10, 6))
sns.histplot(playstore_df['Rating'], bins=20, kde=True)
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(playstore_df['Installs'], playstore_df['revenue'], alpha=0.5)
plt.title('Revenue vs Installs')
plt.xlabel('Installs')
plt.ylabel('Revenue')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Price', data=playstore_df)
plt.xticks(rotation=90)
plt.title('App Prices by Category')
plt.show()

sentiment_counts = reviews_df.groupby('App')['Sentiment'].value_counts().unstack().fillna(0)
sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Sentiment Count for Each App')
plt.xlabel('App')
plt.ylabel('Sentiment Count')
plt.show()

print(playstore_df.head())
