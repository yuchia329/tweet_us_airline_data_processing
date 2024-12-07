import part1
import part5


def main():
    df = part1.load_df()
    print('Part1 A1:')
    part1.airline_group_count(df)
    print('Part1 A2:')
    part1.airline_group_airline_sentiment_negativereason(df)
    print('Part1 A3:')
    part1.airline_group_tweet_len(df)
    print('Part1 A4:')
    part1.airline_group_plot_tweet_len_dist(df)
    print('Part1 B:')
    part1.airline_group_single_plot(df)
    print('Part1 C D:')
    part1.compare_tokenizers(df)

    df = part5.load_df()
    print('Part5.1:')
    part5.get_unique_users(df)
    print('Part5.2:')
    part5.get_active_user_by_airline(df)
    print('Part5.3:')
    part5.drop_missing_location(df)
    print('Part5.4:')
    part5.parse_tweet_created(df)
    print('Part5.5:')
    part5.find_Philadelphia(df)
    print('Part5.6:')
    part5.get_upper_sentiment_confidence(df)


if __name__ == "__main__":
    main()
