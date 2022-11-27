import pickle
import sys
from typing import Optional, List

import pandas as pd
from autogluon.common.loaders import load_pkl

from ..loaders import Paths

from .sim_utils import get_dataset_to_tid_dict, get_dataset_name_to_tid_dict, filter_datasets
from .tabular_predictions import TabularPicklePredictions, TabularPicklePerTaskPredictions
from ..utils.rank_utils import RankScorer


class ZeroshotSimulatorContext:
    def __init__(
            self, 
            df_results_by_dataset: pd.DataFrame,
            df_results_by_dataset_automl: pd.DataFrame,
            df_raw: pd.DataFrame, 
            folds: List[int]
    ):
        """
        Encapsulates results evaluated on multiple base models/datasets/folds.
        :param df_results_by_dataset: results of base models on multiple datasets/folds
        :param df_results_by_dataset_automl: results of automl systems by multiple datasets/folds
        :param df_raw: 
        :param folds: List of folds to be considered in a list of integers
        """
        self.folds = folds

        self.df_results_by_dataset_vs_automl, \
        self.df_raw, \
        self.dataset_name_to_tid_dict, \
        self.dataset_to_tid_dict, \
        self.dataset_name_to_fold_dict, \
        self.unique_dataset_folds, \
        self.unique_datasets, \
        self.rank_scorer_vs_automl = self.align_valid_folds(
            df_results_by_dataset=df_results_by_dataset,
            df_results_by_dataset_automl=df_results_by_dataset_automl,
            df_raw=df_raw,
            folds=folds,
        )
        self.dataset_parent_to_fold_map = self._compute_dataset_parent_to_fold_map()

        tmp = self.df_results_by_dataset_vs_automl[['dataset', 'tid', 'problem_type']]
        self.dataset_to_problem_type_dict = tmp[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze().to_dict()
        self.tid_to_problem_type_dict = tmp[['tid', 'problem_type']].drop_duplicates().set_index(
            'tid').squeeze().to_dict()

    def _compute_dataset_parent_to_fold_map(self) -> dict:
        """
        Returns the mapping of dataset parent to dataset fold names.
        For example:
        {
            'DATASET_NAME': ['DATASET_NAME_1', 'DATASET_NAME_2', ..., 'DATASET_NAME_10'],
            ...,
        }

        """
        dataset_parent_to_fold_map = dict()
        for d in self.unique_dataset_folds:
            dataset_parent = self.dataset_name_to_tid_dict[d]
            if dataset_parent in dataset_parent_to_fold_map:
                dataset_parent_to_fold_map[dataset_parent].append(d)
            else:
                dataset_parent_to_fold_map[dataset_parent] = [d]
        for d in dataset_parent_to_fold_map:
            dataset_parent_to_fold_map[d] = sorted(dataset_parent_to_fold_map[d])
        return dataset_parent_to_fold_map

    @staticmethod
    def align_valid_folds(df_results_by_dataset, df_results_by_dataset_automl, df_raw, folds):
        df_results_by_dataset = df_results_by_dataset[df_results_by_dataset['fold'].isin(folds)]
        unique_dataset_folds_set = set(list(df_results_by_dataset['dataset'].unique()))
        df_results_by_dataset_automl = df_results_by_dataset_automl[
            df_results_by_dataset_automl['dataset'].isin(unique_dataset_folds_set)]

        unique_dataset_folds_set = set(list(df_results_by_dataset_automl['dataset'].unique()))
        df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                        df_raw=df_raw,
                                                        datasets=unique_dataset_folds_set)

        a = df_results_by_dataset[['tid', 'fold']].drop_duplicates()
        a = a[a['fold'].isin(folds)]
        b = a['tid'].value_counts()
        b = b[b == len(folds)]
        unique_datasets = list(b.index)

        dataset_name_to_fold_dict = df_results_by_dataset[['dataset', 'fold']].drop_duplicates().set_index('dataset')[
            'fold'].to_dict()

        dataset_name_to_tid_dict = get_dataset_name_to_tid_dict(df_raw=df_raw)
        unique_dataset_folds = []
        unique_datasets_set = set(unique_datasets)
        for dataset in unique_dataset_folds_set:
            if dataset_name_to_tid_dict[dataset] in unique_datasets_set:
                unique_dataset_folds.append(dataset)
        unique_dataset_folds = sorted(unique_dataset_folds)
        unique_dataset_folds_set = set(unique_dataset_folds)

        df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                        df_raw=df_raw,
                                                        datasets=unique_dataset_folds_set)

        dataset_name_to_tid_dict = get_dataset_name_to_tid_dict(df_raw=df_raw)
        dataset_to_tid_dict = get_dataset_to_tid_dict(df_raw=df_raw)

        automl_error_dict = {}
        for i, dataset in enumerate(unique_dataset_folds):
            automl_error_list = sorted(
                list(df_results_by_dataset_automl[df_results_by_dataset_automl['dataset'] == dataset]['metric_error']))
            automl_error_dict[dataset] = automl_error_list

        rank_scorer_vs_automl = RankScorer(df_results_by_dataset=df_results_by_dataset_automl,
                                           datasets=unique_dataset_folds)
        df_results_by_dataset_vs_automl = df_results_by_dataset.copy()
        df_results_by_dataset_vs_automl['rank'] = [rank_scorer_vs_automl.rank(r[1], r[0]) for r in
                                                   zip(df_results_by_dataset_vs_automl['metric_error'],
                                                       df_results_by_dataset_vs_automl['dataset'])]

        return (
            df_results_by_dataset_vs_automl,
            df_raw,
            dataset_name_to_tid_dict,
            dataset_to_tid_dict,
            dataset_name_to_fold_dict,
            unique_dataset_folds,
            unique_datasets,
            rank_scorer_vs_automl,
        )

    def print_info(self):
        out = '====== Zeroshot Simulator Context Info ======\n'
        out += f'# Configs: {len(self.get_configs())}\n'
        out += f'# Datasets: {len(self.unique_datasets)}\n'
        out += f'# Folds: {len(self.folds)}\n'
        out += f'Folds: {self.folds}\n'
        out += f'# Folds*Datasets: {len(self.unique_dataset_folds)}\n'
        out += '=============================================\n'
        print(out)

    def get_datasets(self, problem_type=None):
        datasets = self.unique_datasets
        if problem_type is not None:
            datasets = [dataset for dataset in datasets if self.tid_to_problem_type_dict[dataset] == problem_type]
        return datasets

    def get_dataset_folds(self, datasets: Optional[List[str]] = None, problem_type: Optional[str] = None) -> List[str]:
        """
        :param datasets: a list of dataset parent names, only return folds that have a parent in this list
        :param problem_type: a problem type from AutoGluon in "multiclass", "binary", ...
        :return: List of datasets-folds formatted as `['359987_8', '359933_3', ...]` where the dataset is encoded before
        the "_" and the fold after.
        # Todo/Note it might be clearer to add a column fold in the dataframe and return List[Tuple[str, int]] with
        tuples of dataset/fold.
        """
        if datasets is not None:
            dataset_folds = self._get_dataset_folds_from_datasets(datasets=datasets)
        else:
            dataset_folds = self.unique_dataset_folds
        if problem_type is not None:
            dataset_folds = [dataset for dataset in dataset_folds if self.dataset_to_problem_type_dict[dataset] == problem_type]
        return dataset_folds

    def _get_dataset_folds_from_datasets(self, datasets: List[str]):
        dataset_folds = []
        for d in datasets:
            dataset_folds += self.dataset_parent_to_fold_map[d]
        return dataset_folds

    def get_configs(self) -> list:
        """Return all valid configs"""
        return list(self.df_results_by_dataset_vs_automl['framework'].unique())

    def load_groundtruth(self, path_gt: str) -> dict:
        zeroshot_gt = load_pkl.load(path_gt)
        zeroshot_gt = {k: v for k, v in zeroshot_gt.items() if k in self.dataset_to_tid_dict}
        zeroshot_gt = {self.dataset_to_tid_dict[k]: v for k, v in zeroshot_gt.items()}
        return zeroshot_gt

    def load_pred(self, path_pred_proba: str, lazy_format: bool = False) -> dict:
        print('Loading zeroshot...')
        # NOTE: This file is BIG (17 GB)
        cls = TabularPicklePerTaskPredictions if lazy_format else TabularPicklePredictions
        if lazy_format:
            # convert to lazy format if format not already available
            self.convert_lazy_format()
        zeroshot_pred_proba = cls.load(path_pred_proba)
        for k in zeroshot_pred_proba.datasets:
            if k not in self.dataset_to_tid_dict:
                zeroshot_pred_proba.remove_dataset(k)
        # rename dataset to dataset-ids, eg. 'abalone' is mapped to 359944.0
        zeroshot_pred_proba.rename_datasets({
            k: self.dataset_to_tid_dict[k]
            for k in zeroshot_pred_proba.datasets
        })
        return zeroshot_pred_proba

    @staticmethod
    def convert_lazy_format(override_if_already_exists: bool = False):
        new_filename = Paths.all_v3_results_root / "zeroshot_pred_per_task"
        if not new_filename.exists() or override_if_already_exists:
            print(f"lazy format folder {new_filename} not found or override option set to True, "
                  f"converting to lazy format. It should take less than 3 min.")
            preds = TabularPicklePredictions.load(str(Paths.all_v3_results_root / "zeroshot_pred_proba_2022_10_13_zs.pkl"))
            preds_npy = TabularPicklePerTaskPredictions.from_dict(preds.pred_dict, output_dir=str(new_filename))

    @staticmethod
    def minimize_memory_zeroshot_pred_proba(zeroshot_pred_proba: dict, configs: list):
        """
        Minimizes memory usage of zeroshot_pred_proba by popping all model keys not in the input configs list.

        Note: Performs inplace edits.
        """
        if configs is None:
            return zeroshot_pred_proba
        size_bytes = sys.getsizeof(pickle.dumps(zeroshot_pred_proba, protocol=4))
        print(f'OLD zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
        task_names = list(zeroshot_pred_proba.keys())
        configs = set(configs)
        for t in task_names:
            available_folds = list(zeroshot_pred_proba[t].keys())
            for f in available_folds:
                model_keys = list(zeroshot_pred_proba[t][f]['pred_proba_dict_val'].keys())
                for k in model_keys:
                    if k not in configs:
                        zeroshot_pred_proba[t][f]['pred_proba_dict_val'].pop(k)
                        zeroshot_pred_proba[t][f]['pred_proba_dict_test'].pop(k)
        size_bytes = sys.getsizeof(pickle.dumps(zeroshot_pred_proba, protocol=4))
        print(f'NEW zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
        return zeroshot_pred_proba
