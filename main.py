from processor import DataProcessor


def main():
    processor = DataProcessor()
    processor.load_csv_to_db('data/train.csv', 'train_data')
    processor.load_csv_to_db('data/ideal.csv', 'ideal_data')
    processor.load_csv_to_db('data/test.csv', 'test_data')

    train_df = processor.get_data('train_data')
    ideal_df = processor.get_data('ideal_data')
    test_df = processor.get_data('test_data')

    best_fits = processor.find_best_fit(train_df, ideal_df)
    results_df = processor.map_test_data(test_df, best_fits, train_df, ideal_df)
    processor.save_results_to_db(results_df, 'results_data')

    processor.visualize_data(train_df, test_df, ideal_df, best_fits, results_df)


if __name__ == "__main__":
    main()