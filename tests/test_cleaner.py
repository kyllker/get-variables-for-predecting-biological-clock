import os
import sys
import numpy as np
import pandas as pd
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.Preprocessors.cleaner import Cleaner


class TestCleaner:
    def setup_class(self):
        self.seed = 42
        self.cleaner_object = Cleaner(self.seed)

        self.df = pd.DataFrame([[1, 1], [1, 2]], columns=['A', 'B'])

        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])

    def test_build_a_cleaner_object(self):
        assert isinstance(self.cleaner_object, Cleaner)

    def test__checker_filter_valid_columns(self):
        dataframe_desired_columns = self.cleaner_object.filter_desired_columns(self.df, [0])
        assert dataframe_desired_columns.equals(pd.DataFrame([[1], [1]], columns=['A']))

    def test__checker_filter_not_valid_columns(self):
        dataframe_desired_columns = self.cleaner_object.filter_desired_columns(self.df, [0, 2])
        assert dataframe_desired_columns.equals(self.df)

    def test__drop_duplicate_columns(self):
        df2 = self.df.copy()
        df2['C'] = [1, 1]
        dataframe_drop_duplicate_columns = self.cleaner_object.remove_duplicate_columns(df2)
        assert self.df.equals(dataframe_drop_duplicate_columns)

    def test__drop_duplicate_columns_when_there_are_not(self):
        df2 = self.df.copy()
        dataframe_drop_duplicate_columns = self.cleaner_object.remove_duplicate_columns(df2)
        assert self.df.equals(dataframe_drop_duplicate_columns)

    def test__drop_constant_columns(self):
        df2 = self.df.copy()
        df2['C'] = [1, 3]
        dataframe_drop_duplicate_columns = self.cleaner_object.remove_constant_columns(df2)
        assert dataframe_drop_duplicate_columns.equals(pd.DataFrame([[1, 1], [2, 3]], columns=['B', 'C']))

    def test__drop_constant_columns_when_there_are_not(self):
        df2 = pd.DataFrame([[1, 1], [2, 3]], columns=['B', 'C'])
        dataframe_drop_constant_columns = self.cleaner_object.remove_constant_columns(df2)
        assert df2.equals(dataframe_drop_constant_columns)

    # def test__convert_to_numerical_values_column_with_two_different_values(self):
    #     dataframe_convert_numerical_columns = \
    #         self.cleaner_object.convert_to_numerical_values_column_with_two_different_values(self.df_no_numerical)
    #     print(dataframe_convert_numerical_columns)
    #     df_numeric = pd.DataFrame([[0, 0], [1, 1]], columns=['A_a1_c0', 'B_b1_d0'])
    #     assert np.array_equal(dataframe_convert_numerical_columns.to_numpy(), df_numeric.to_numpy())
    #
    # def test__cleaner_normalize_dataframe(self):
    #     df2 = pd.DataFrame([[1, 1], [2, 3]], columns=['B', 'C'])
    #     dataframe_drop_constant_columns = self.cleaner_object.normalize_dataframe(df2)
    #     df_res = pd.DataFrame([[0, 0], [1, 1]], columns=['B', 'C'])
    #     assert np.array_equal(dataframe_drop_constant_columns.to_numpy(), df_res.to_numpy())
    #
    # def test__predict(self):
    #     df = pd.DataFrame([['id1', 'a', 'b', 'w', 'a', 'f'], ['id2', 'c', 'd', 'w', 'c', 'd'],
    #                        ['id3', 'a', 'j', 'w', 'a', 'k']],
    #                       columns=['ID_Muestra', 'A', 'B', 'C', 'D', 'E'])
    #     df_res = self.cleaner_object.predict(df, ['ID_Muestra', 'A', 'B', 'C', 'D'])
    #     df_pred = pd.DataFrame([['id1', 0, 1, 0], ['id2', 1, 0, 0], ['id3', 0, 0, 1]],
    #                            columns=['ID_Muestra', 'A', 'B_b', 'B_j'])
    #     assert np.array_equal(df_res.to_numpy(), df_pred.to_numpy())
