import json
import pickle
import shutil
import sys
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers.save_pkl import save as save_pkl

from autogluon_zeroshot.simulation.tabular_predictions import TabularPicklePredictions


def is_dense(self) -> bool:
    """
    Return True if all datasets have all models
    """
    models_dense = self.models
    models_sparse = self.get_models(present_in_all=False)
    return set(models_dense) == set(models_sparse)

def get_models(self, present_in_all=False) -> List[str]:
    """
    Gets all valid models

    :param present_in_all:
        If True, only returns models present in every dataset (dense)
        If False, returns every model that appears in at least 1 dataset (sparse)
    """
    if present_in_all:
        return self.models
    models = set()
    for dataset in self.datasets:
        for fold in self.folds:
            for split in ["pred_proba_dict_val", "pred_proba_dict_test"]:
                for k in self.pred_dict[dataset][fold][split].keys():
                    models.add(k)
    # returns models that appears in all lists, eg that are available for all datasets, folds and splits
    return list(models)

def restrict_models(self, models: List[str]):
    # FIXME: Make this the default restrict_models logic. Implement this sanity check in lazy mode
    models_present = self.get_models(present_in_all=False)
    for m in models:
        assert m in models_present, f"cannot restrict {m} which is not in available models {models_present}."
    self._restrict_models(models)

def _restrict_models(self, models: List[str]):
    size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
    print(f'OLD zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
    task_names = self.datasets
    configs = set(models)
    for t in task_names:
        available_folds = list(self.pred_dict[t].keys())
        for f in available_folds:
            model_keys = list(self.pred_dict[t][f]['pred_proba_dict_val'].keys())
            for k in model_keys:
                if k not in configs:
                    self.pred_dict[t][f]['pred_proba_dict_val'].pop(k)
                    self.pred_dict[t][f]['pred_proba_dict_test'].pop(k)
    size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
    print(f'NEW zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')

def get_datasets_with_models(self, models: List[str]) -> List[str]:
    """
    Get list of datasets that have results for all input models
    """
    task_names = self.datasets
    configs = set(models)
    valid_tasks = []
    for t in task_names:
        is_valid = True
        available_folds = list(self.pred_dict[t].keys())
        for f in available_folds:
            model_keys = self.pred_dict[t][f]['pred_proba_dict_val'].keys()
            for m in configs:
                if m not in model_keys:
                    is_valid = False
        if is_valid:
            valid_tasks.append(t)
    return valid_tasks


# TODO: Implement in lazy
def is_empty(self) -> bool:
    """
    Return True if no models or datasets exist
    """
    return len(self.datasets) == 0 or len(self.get_models(present_in_all=False)) == 0


def restrict_datasets(self, datasets: List[str]):
    task_names = self.datasets
    task_names_set = set(task_names)
    for d in datasets:
        if d not in task_names_set:
            raise AssertionError(f'Trying to remove dataset that does not exist! ({d})')
    valid_datasets_set = set(datasets)
    size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
    print(f'Restricting Datasets... (Shrinking from {len(self.datasets)} -> {len(valid_datasets_set)} datasets)')
    print(f'OLD zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
    for t in task_names:
        if t not in datasets:
            self.pred_dict.pop(t)
    size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
    print(f'NEW zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')

# TODO: Implement in lazy
def force_to_dense(zeroshot_pred_proba: TabularPicklePredictions, prune_method: str = 'dataset', assert_not_empty: bool = True):
    """
    Force the pred dict to contain only dense results (no missing result for any dataset/model)

    :param prune_method:
        If 'dataset', prunes any dataset that doesn't contain results for all models
        If 'model', prunes any model that doesn't have results for all datasets
    """

    if prune_method == 'dataset':
        valid_models = zeroshot_pred_proba.get_models(present_in_all=False)
        valid_datasets = zeroshot_pred_proba.get_datasets_with_models(models=valid_models)
        self.restrict_datasets(datasets=valid_datasets)
    elif prune_method == 'model':
        valid_models = self.get_models(present_in_all=True)
        self.restrict_models(models=valid_models)
    assert self.is_dense()
    if assert_not_empty:
        assert not self.is_empty()
