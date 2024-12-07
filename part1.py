import pandas as pd
import plot
import math
import matplotlib.pyplot as plt
import re
import nltk
import random


def load_df():
    df = pd.read_csv("Tweets.csv")
    df = df[["airline_sentiment", "airline_sentiment_confidence", "negativereason", "negativereason_confidence",
            "airline", "airline_sentiment_gold", "negativereason_gold", "text"]]
    return df


def airline_group_count(df):
    grouped = df.groupby('airline')['airline_sentiment'].count()
    print(f'total count of data samples: {grouped}')
    print("\n-------------------------END----------------------\n")


def airline_group_airline_sentiment_negativereason(df):
    columns_to_analyze = ['airline_sentiment', 'negativereason']
    unique_arilines = df['airline'].unique()

    for airline in unique_arilines:
        df_airline = df[df['airline'] == airline]

        # Create a summary DataFrame
        summary = {
            column: column_stats(df_airline, column, airline)
            for column in columns_to_analyze
        }

        summary_df = pd.DataFrame(
            summary, index=['Airline', 'Unique Count', 'Most Frequent Value', 'Frequency'])

        # Display the result
        print(summary_df)
        print("\n-------------------------END----------------------\n")


# Function to compute statistics
def column_stats(df, column, airline):
    unique_count = df[column].nunique()
    most_frequent_value = df[column].mode()[0]
    frequency = df[column].value_counts().iloc[0]
    return airline, unique_count, most_frequent_value, frequency


def airline_group_tweet_len(df):
    unique_arilines = df['airline'].unique()
    for airline in unique_arilines:
        df_airline = df[df['airline'] == airline]
        df['tweet_length'] = df_airline['text'].str.len()
        shortest_tweet_length = df['tweet_length'].min()
        longest_tweet_length = df['tweet_length'].max()
        print(f"Airline: {airline}, Shortest tweet length: {
              shortest_tweet_length}")
        print(f"Airline: {airline}, Longest tweet length: {
              longest_tweet_length}")
        print('\n')
    print("\n-------------------------END----------------------\n")


def airline_group_plot_tweet_len_dist(df):
    unique_arilines = df['airline'].unique()
    for airline in unique_arilines:
        df_airline = df[df['airline'] == airline]
        df_airline.loc[:, 'tweet_length'] = df_airline['text'].str.len()
        max_length = df_airline['tweet_length'].max()
        max_bin = math.ceil(max_length / 5) * 5
        bins = list(range(0, max_bin + 1, 5))
        plot.plot_histogram(df_airline, bins, airline)


def airline_group_single_plot(df):
    df["airline_sentiment"] = df["airline_sentiment"].astype("category")
    unique_arilines = df['airline'].unique()
    sentiments = ["positive", "neutral", "negative"]

    colors = {"positive": "green", "neutral": "blue", "negative": "red"}

    # Create the plot
    plt.clf()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, airline in enumerate(unique_arilines):
        # Filter the data for the current airline
        airline_data = df[df["airline"] == airline]

        # Plot the histogram for sentiment distribution
        sentiment_counts = airline_data["airline_sentiment"].value_counts().reindex(
            sentiments, fill_value=0)
        axes[i].bar(sentiments, sentiment_counts, color=[colors[s]
                    for s in sentiments], edgecolor="black")

        # Set the title and labels for the subplot
        axes[i].set_title(f"Sentiment Distribution - {airline}")
        axes[i].set_xlabel("Sentiment")
        axes[i].set_ylabel("Count")

    # Remove unused subplots
    for j in range(len(unique_arilines), len(axes)):
        fig.delaxes(axes[j])

    # Display the plot
    plt.savefig("Part1B.jpg")
    print(f'Single grid-like plot saved in local')
    print("\n-------------------------END----------------------\n")


def custom_tokenizer(text):
    """
    Custom tokenizer that tokenizes text based on specific rules.
    """
    # Rule 1: lowercase
    text = text.lower()

    # Rule 2: Split by whitespace and punctuation marks (while preserving punctuation as tokens)
    tokens = re.split(r'(\s+|[,.!?;:"\'()\-])', text)

    # Rule 3: Remove empty tokens and unnecessary whitespace-only tokens
    tokens = [token.strip() for token in tokens if token.strip()]

    # Rule 4: Split contractions (e.g., "can't" -> "ca", "n't")
    expanded_tokens = []
    for token in tokens:
        contraction_match = re.match(r"(\w+)('t|'ll|'re|'ve|'s|'d|'m)", token)
        if contraction_match:
            expanded_tokens.append(contraction_match.group(1))
            expanded_tokens.append(contraction_match.group(2))
        else:
            expanded_tokens.append(token)
    tokens = expanded_tokens

    # Rule 5: Handle special tokens such as URLs or email addresses
    special_token_pattern = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
    final_tokens = []
    for token in tokens:
        if special_token_pattern.match(token):
            final_tokens.append(token)
        else:
            final_tokens.extend(re.split(r'(\s+|[,.!?;:"\'()\-])', token))
    tokens = [token.strip() for token in final_tokens if token.strip()]

    # Rule 6: Separate @ and # from tokens
    final_tokens = []
    print(tokens)
    for token in tokens:
        if re.match(r'([@#])(\w+)', token):
            match = re.match(r'([@#])(\w+)', token)
            print(match[1])
            final_tokens.append(match.group(1))
            final_tokens.append(match.group(2))
        else:
            final_tokens.append(token)

    return final_tokens


def downloadNLTK():
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def compare_tokenizers(df):

    text = df["text"]
    total_rows = df.shape[0]
    random.seed(42)
    random_index = [random.randint(0, total_rows) for _ in range(5)]
    print(random_index)
    texts = [text.iloc[index] for index in random_index]
    # Test the custom tokenizer
    downloadNLTK()
    for index, text in enumerate(texts):
        custom_tokens = set(custom_tokenizer(text))
        nltk_tokens = set(nltk.word_tokenize(text))
        intersection = custom_tokens.intersection(nltk_tokens)
        difference = custom_tokens.symmetric_difference(nltk_tokens)
        print(f"sentence index: {index}")
        print(f"sentence: {text}")
        print("custom_tokens: ", custom_tokens)
        print("nltk_tokens: ", nltk_tokens)
        print("Intersection: ", intersection)
        print("Difference: ", difference)
        print("----------------------------------\n")

    x = random.randint(1, 10)
    print(x)
    print("\n-------------------------END----------------------\n")
