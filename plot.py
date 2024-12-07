import matplotlib.pyplot as plt


def plot_histogram(df, bins, airline):
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.hist(df['tweet_length'], bins=bins, edgecolor='black', alpha=0.75)
    plt.title(f"Distribution of Tweet Lengths for {airline}", fontsize=14)
    plt.xlabel("Tweet Length", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    name = f"Part1A-Histogram-{airline}"
    plt.gcf().canvas.manager.set_window_title(name)
    plt.savefig(name)
    print(f'Histogram {name} saved in local')
    print("\n-------------------------END----------------------\n")


def plot_sentiment(df, airline):
    colors = {"positive": "green", "neutral": "blue", "negative": "red"}

    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()
