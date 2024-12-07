import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_df():
    df = pd.read_csv("Tweets.csv")
    return df


def get_unique_users(df):
    unique_users = df['name'].unique()
    print(unique_users)

    # Calculate TF-IDF for each user and find top-5 words
    user_top_words = {}

    for user in unique_users:
        user_tweets = df[df['name'] == user]['text']
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(user_tweets)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores for each word across all tweets
        word_scores = tfidf_matrix.sum(axis=0).A1
        word_scores_dict = dict(zip(feature_names, word_scores))

        # Get top-5 words by TF-IDF score
        top_words = sorted(word_scores_dict.items(),
                           key=lambda x: x[1], reverse=True)[:5]
        user_top_words[user] = [word for word, score in top_words]

    result_df = pd.DataFrame.from_dict(user_top_words, orient='index', columns=[
                                       'Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
    print(result_df)
    print("\n-------------------------END----------------------\n")


def get_active_user_by_airline(df):
    # Find the most active user for each airline
    most_active_users = df.groupby("airline")["name"].apply(
        lambda x: x.value_counts().idxmax())

    # Get the details of the most active users
    most_active_user_details = df[df["name"].isin(most_active_users.values)][
        ["airline", "name", "text", "tweet_location", "airline_sentiment"]
    ]

    # Display the result
    print(most_active_user_details)
    print("\n-------------------------END----------------------\n")


def drop_missing_location(df):
    # Find the number of missing values in 'tweet_location' and 'user_timezone'
    missing_values = df[["tweet_location", "user_timezone"]].isnull().sum()
    print("Number of missing values:")
    print(missing_values)

    # Drop rows with missing values in 'tweet_location' or 'user_timezone'
    df_cleaned = df.dropna(subset=["tweet_location", "user_timezone"])

    # Display the cleaned DataFrame
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    print("\n-------------------------END----------------------\n")


def parse_tweet_created(df):
    # Check the type of the 'tweet_created' field
    print(f"Initial type of 'tweet_created': {df['tweet_created'].dtype}")

    # Parse 'tweet_created' as datetime
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')

    # Check the type after parsing
    print(f"Type after parsing 'tweet_created': {df['tweet_created'].dtype}")
    df = df[['tweet_created', 'user_timezone']]
    # Display the DataFrame
    print("\nParsed DataFrame:")
    print(df)
    print("\n-------------------------END----------------------\n")


def find_Philadelphia(df):
    # Normalize the tweet_location column
    def normalize_location(location):
        location = str(location)
        return location.lower().replace(" ", "").replace(",", "")

    # Apply normalization
    df['normalized_location'] = df['tweet_location'].map(normalize_location)

    # Find unique variations of Philadelphia
    possible_mistype = ['philadelphia', 'philadelpia', 'philadephia', 'phila', 'phile',
                        'phel' 'philly', 'phillies', 'filly', 'filadelphia', 'filadelfia']
    philadelphia_variations = df[df['normalized_location'].str.contains(
        '|'.join(possible_mistype))]['tweet_location'].unique()

    # Count tweets from Philadelphia
    philadelphia_count = df['normalized_location'].str.contains(
        '|'.join(possible_mistype)).sum()

    # Display results
    print(f"Total number of tweets from Philadelphia: {philadelphia_count}")
    print("Different spellings of Philadelphia:")
    print(philadelphia_variations)
    print("\n-------------------------END----------------------\n")


def get_upper_sentiment_confidence(df):
    # Create a subset where airline_sentiment_confidence > 0.6
    subset_df = df[df["airline_sentiment_confidence"] > 0.6]

    # Save the subset to a CSV file
    output_file_path = "subset_dataset.csv"
    subset_df.to_csv(output_file_path, index=False)

    print(f"Subset dataset saved to {output_file_path}")
    print("\n-------------------------END----------------------\n")
