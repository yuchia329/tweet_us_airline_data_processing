import re
from nltk.stem import WordNetLemmatizer
from html import unescape
from part1 import load_df, downloadNLTK


def remove_mentions(text):
    # Rule 1: Remove mentions (e.g., @united)
    return re.sub(r'@\w+', '', text)


def remove_currency(text):
    # Rule 2: Remove currency symbols and amounts (e.g., $19.90)
    text = re.sub(r'\$\d+(\.\d+)?', '', text)
    return text


def remove_emails(text):
    # Rule 3: Remove email addresses (e.g., jane.doe@email.com)
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    return text


def remove_emojis(text):
    # Rule 4: Remove emojis
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    return text


def replace_html_chars(text):
    # Rule 5: Replace HTML escaped characters with their actual meaning
    text = unescape(text)
    return text


def normalize_punctuation(text):
    # Rule 6: Normalize excessive punctuation (e.g., "!!!!" -> "!")
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    return text


def normalize_dates_times(text):
    # Rule 7: Normalize times & dates (replace them with a placeholder)
    # Times like 2:10pm or 7:00 AM
    text = re.sub(r'\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?\b', '<TIME>', text)
    text = re.sub(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
                  '<DATE>', text)  # Dates like 2/24 or 6/30/2020
    return text


def remove_urls(text):
    # Rule 8: Remove URLs (e.g., http://t.co/NfAQHhr09j)
    text = re.sub(r'http[s]?://\S+', '', text)
    return text


def remove_hashtags(text):
    # Rule 9: Remove hashtags (keep the word but drop the "#")
    text = re.sub(r'#(\w+)', r'\1', text)
    return text


def remove_non_alphanumeric(text):
    # Rule 10: Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text


def normalize_spaces(text):
    # Rule 11: Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text):
    # Rule 12: Perform lemmatization on each word
    downloadNLTK()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)
    return text


RULES = {
    "1": remove_mentions,
    "2": remove_currency,
    "3": remove_emails,
    "4": remove_emojis,
    "5": replace_html_chars,
    "6": normalize_punctuation,
    "7": normalize_dates_times,
    "8": remove_urls,
    "9": remove_hashtags,
    "10": remove_non_alphanumeric,
    "11": normalize_spaces,
    "12": lemmatize_text
}


def clean_text(text, order):
    for index in order:
        if index in RULES:  # Apply the rule only if it exists in the mapping
            text = RULES[index](text)
    return text


def preprocess_data(rule_order=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    # Apply the cleaning function to the text column
    df = load_df()
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(
        x, order=rule_order))

    # Drop duplicate rows based on the cleaned_text and sentiment
    df = df.drop_duplicates(subset=['cleaned_text', 'airline_sentiment'])

    # Remove rows with empty tweets after cleaning
    df = df[df['cleaned_text'].str.strip() != '']
    return df
