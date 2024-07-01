import unittest
from processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.processor.load_csv_to_db('data/train.csv', 'train_data')
        self.processor.load_csv_to_db('data/ideal.csv', 'ideal_data')
        self.processor.load_csv_to_db('data/test.csv', 'test_data')

    def test_get_data(self):
        train_df = self.processor.get_data('train_data')
        self.assertEqual(len(train_df), 400)  # assuming 400 rows in train.csv

    def test_find_best_fit(self):
        train_df = self.processor.get_data('train_data')
        ideal_df = self.processor.get_data('ideal_data')
        best_fits = self.processor.find_best_fit(train_df, ideal_df)
        self.assertEqual(len(best_fits), 4)  # there are 4 training columns

    def test_map_test_data(self):
        train_df = self.processor.get_data('train_data')
        ideal_df = self.processor.get_data('ideal_data')
        test_df = self.processor.get_data('test_data')
        best_fits = self.processor.find_best_fit(train_df, ideal_df)
        results_df = self.processor.map_test_data(test_df, best_fits, train_df, ideal_df)
        self.assertEqual(len(results_df), len(test_df))  # should have same number of rows as test_df

    def test_save_results_to_db(self):
        test_df = self.processor.get_data('test_data')
        best_fits = self.processor.find_best_fit(self.processor.get_data('train_data'),
                                                 self.processor.get_data('ideal_data'))
        results_df = self.processor.map_test_data(test_df, best_fits, self.processor.get_data('train_data'),
                                                  self.processor.get_data('ideal_data'))
        self.processor.save_results_to_db(results_df, 'results_data')
        results_from_db = self.processor.get_data('results_data')
        self.assertEqual(len(results_from_db), len(results_df))  # check if results saved correctly


if __name__ == "__main__":
    unittest.main()