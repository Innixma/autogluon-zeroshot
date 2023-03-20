from typing import List

import numpy as np
import pandas as pd


class RankScorer:
    def __init__(self,
                 df_results_by_dataset: pd.DataFrame,
                 datasets: List[str],
                 metric_error_col: str = 'metric_error',
                 dataset_col: str = 'dataset',
                 framework_col: str = 'framework',
                 pct: bool = False,
                 ):
        """
        :param df_results_by_dataset: Dataframe of method performance containing columns `metric_error_col`,
        `dataset_col` and `framework_col`.
        :param datasets: datasets to consider
        :param pct: whether or not to display the returned rankings in percentile form.
        """
        assert all(col in df_results_by_dataset for col in [metric_error_col, dataset_col, framework_col])
        all_datasets = set(df_results_by_dataset[dataset_col].unique())
        for dataset in datasets:
            assert dataset in all_datasets, f"dataset {dataset} not present in passed evaluations"
        self.pct = pct
        df_pivot = df_results_by_dataset.pivot_table(values=metric_error_col, index=dataset_col, columns=framework_col)
        df_pivot.values.sort(axis=1)
        self.error_dict = {dataset: df_pivot.loc[dataset] for dataset in datasets}

    def rank(self, dataset: str, error: float) -> float:
        baseline_scores = self.error_dict[dataset]

        rank = np.searchsorted(baseline_scores, error)  # value in [0, num-baselines)]
        if self.pct:
            rank /= len(baseline_scores)
        else:
            rank += 1  # value in [1, 1 + num-baselines)]
        return rank
