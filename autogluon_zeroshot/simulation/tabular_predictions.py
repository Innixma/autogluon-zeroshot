import json
import pickle
import shutil
import sys
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers.save_pkl import save as save_pkl

# dictionary mapping the config name to predictions for a given dataset fold split
ConfigPredictionsDict = Dict[str, np.array]

# dictionary mapping a particular fold of a dataset (a task) to split to config name to predictions
TaskPredictionsDict = Dict[str, ConfigPredictionsDict]

# dictionary mapping the folds of a dataset to split to config name to predictions
DatasetPredictionsDict = Dict[int, TaskPredictionsDict]

# dictionary mapping dataset to fold to split to config name to predictions
TabularPredictionsDict = Dict[str, DatasetPredictionsDict]


class TabularModelPredictions:
    """
    Class that allows to query offline predictions.
    """

    def predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        """
        :param dataset:
        :param fold:
        :param splits: split to consider values must be in 'val' or 'test'
        :param models: list of models to be evaluated, by default uses all models available
        :return: for each split, a tensor with shape (num_models, num_points) for regression and
        (num_models, num_points, num_classes) for classification corresponding the predictions of the model.
        """
        if splits is None:
            splits = ['val', 'test']
        for split in splits:
            assert split in ['val', 'test']
        assert models is None or len(models) > 0
        return self._predict(dataset, fold, splits, models)

    @property
    def models(self) -> List[str]:
        """
        :return: list of models present in at least one dataset.
        """
        raise NotImplementedError()

    def get_dataset(self, dataset: str) -> DatasetPredictionsDict:
        raise NotImplementedError()

    def get_task(self, dataset: str, fold: int) -> TaskPredictionsDict:
        return self.get_dataset(dataset=dataset)[fold]

    def _check_dataset_exists(self, dataset: str) -> bool:
        """
        Simple implementation to check if a dataset exists.
        Consider implementing optimized version in inheriting classes if this is time-consuming.
        """
        return dataset in self.datasets

    def _check_task_exists(self, dataset: str, fold: int) -> bool:
        """
        Simple implementation to check if a task exists.
        Consider implementing optimized version in inheriting classes if this is time-consuming.
        """
        try:
            self.get_task(dataset=dataset, fold=fold)
            return True
        except:
            return False

    def models_available_in_task(self,
                                 *,
                                 task: Optional[TaskPredictionsDict] = None,
                                 dataset: Optional[str] = None,
                                 fold: Optional[int] = None) -> List[str]:
        """
        Get list of valid models for a given task

        Either task must be specified or dataset & fold must be specified.
        """
        if task is None:
            assert dataset is not None
            assert fold is not None
            if self._check_task_exists(dataset=dataset, fold=fold):
                task = self.get_task(dataset=dataset, fold=fold)
            else:
                return []
        else:
            assert dataset is None
            assert fold is None
        models = []
        for split in ["pred_proba_dict_val", "pred_proba_dict_test"]:
            models.append(set(task[split].keys()))
        return sorted(list(set.intersection(*map(set, models))))

    def models_available_in_task_dict(self) -> Dict[str, Dict[int, List[str]]]:
        """Get dict of valid models per task"""
        datasets = self.datasets

        model_fold_dataset_dict = dict()
        for d in datasets:
            dataset_predictions = self.get_dataset(dataset=d)
            model_fold_dataset_dict[d] = dict()
            for f in dataset_predictions:
                model_fold_dataset_dict[d][f] = self.models_available_in_task(task=dataset_predictions[f])
        return model_fold_dataset_dict

    def models_available_in_dataset(self, dataset: str) -> List[str]:
        """Returns the models available on both validation and test splits on all tasks in a dataset"""
        models = []
        folds = self.folds
        dataset_predictions = self.get_dataset(dataset=dataset)
        for fold in folds:
            if fold not in dataset_predictions:
                # if a fold is missing, no models has all folds.
                return []
        for fold in folds:
            task_predictions = dataset_predictions[fold]
            models.append(set(self.models_available_in_task(task=task_predictions)))
        # returns models that appears in all lists, eg that are available for all folds and splits
        return sorted(list(set.intersection(*map(set, models))))

    def restrict_models(self, models: List[str]):
        """
        :param models: restricts the predictions to contain only the list of models given in arguments, useful to
        reduce memory footprint. The behavior depends on the data structure used. For pickle/full data structure,
        the data is immediately sliced. For lazy representation, the data is sliced on the fly when querying predictions.
        """
        models_present = self.models
        for m in models:
            assert m in models_present, f"cannot restrict {m} which is not in available models {models_present}."
        self._restrict_models(models)

    def _restrict_models(self, models: List[str]):
        raise NotImplementedError()

    def restrict_datasets(self, datasets: List[str]):
        for d in datasets:
            assert d in self.datasets, f"cannot restrict {d} which is not in available datasets {self.datasets}."
        self._restrict_datasets(datasets)

    def _restrict_datasets(self, datasets: List[str]):
        for dataset in self.datasets:
            if dataset not in datasets:
                self.remove_dataset(dataset)

    def restrict_folds(self, folds: List[int]):
        folds_cur = self.folds
        for f in folds:
            assert f in folds_cur, f"Trying to restrict to a fold {f} that does not exist! Valid folds: {folds_cur}."
        return self._restrict_folds(folds=folds)

    def _restrict_folds(self, folds: List[int]):
        raise NotImplementedError()

    def remove_dataset(self, dataset: str):
        raise NotImplementedError()

    # TODO: Add is_fold_dense and get_folds_dense to check if all datasets have all folds
    def is_dense(self) -> bool:
        """
        Return True if all datasets have all models
        """
        models_dense = self.get_models(present_in_all=True)
        models_sparse = self.get_models(present_in_all=False)
        return set(models_dense) == set(models_sparse)

    def is_empty(self) -> bool:
        """
        Return True if no models or datasets exist
        """
        return len(self.datasets) == 0 or len(self.get_models(present_in_all=False)) == 0

    def get_models(self, present_in_all=False) -> List[str]:
        """
        Gets all valid models
        :param present_in_all:
            If True, only returns models present in every dataset (dense)
            If False, returns every model that appears in at least 1 dataset (sparse)
        """
        if not present_in_all:
            return self.models
        else:
            return self.get_models_dense()

    def get_models_dense(self) -> List[str]:
        """
        Returns models that appears in all lists, eg that are available for all tasks and splits
        """
        models = []
        for dataset in self.datasets:
            models_in_dataset = set(self.models_available_in_dataset(dataset=dataset))
            models.append(models_in_dataset)
        return list(set.intersection(*map(set, models)))

    def force_to_dense(self, prune_method: str = 'dataset', assert_not_empty: bool = True):
        """
        Force the pred dict to contain only dense results (no missing result for any dataset/model)
        :param prune_method:
            If 'dataset', prunes any dataset that doesn't contain results for all models
            If 'model', prunes any model that doesn't have results for all datasets
        """
        print(f'Forcing {self.__class__.__name__} to dense representation using `prune_method="{prune_method}"...')
        pre_num_models = len(self.models)
        pre_num_datasets = len(self.datasets)
        if prune_method == 'dataset':
            valid_models = self.get_models(present_in_all=False)
            valid_datasets = self.get_datasets_with_models(models=valid_models)
            self.restrict_datasets(datasets=valid_datasets)
        elif prune_method == 'model':
            valid_models = self.get_models(present_in_all=True)
            self.restrict_models(models=valid_models)
        post_num_models = len(self.models)
        post_num_datasets = len(self.datasets)

        print(f'\tPre : datasets={pre_num_datasets} | models={pre_num_models}')
        print(f'\tPost: datasets={post_num_datasets} | models={post_num_models}')
        assert self.is_dense()
        if assert_not_empty:
            assert not self.is_empty()

    def get_datasets_with_models(self, models: List[str]) -> List[str]:
        """
        Get list of datasets that have results for all input models
        """
        datasets = self.datasets
        configs = set(models)
        valid_datasets = []
        for d in datasets:
            models_in_dataset = set(self.models_available_in_dataset(dataset=d))
            is_valid = True
            for m in configs:
                if m not in models_in_dataset:
                    is_valid = False
            if is_valid:
                valid_datasets.append(d)
        return valid_datasets

    @property
    def datasets(self) -> List[str]:
        raise NotImplementedError()

    @property
    def folds(self) -> List[int]:
        raise NotImplementedError()

    @staticmethod
    def _get_task_from_dataset(dataset_predictions: DatasetPredictionsDict, fold: int) -> TaskPredictionsDict:
        return dataset_predictions[fold]

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        """
        :param pred_dict: dictionary mapping dataset to fold to split to config name to predictions
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, filename: str):
        raise NotImplementedError()

    def save(self, filename: str):
        raise NotImplementedError()

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        raise NotImplementedError()


# FIXME: is_dense isn't correct, needs to return 126 datasets from fold 0, not 134
class TabularPicklePredictions(TabularModelPredictions):
    def __init__(self, pred_dict: TabularPredictionsDict):
        self.pred_dict = pred_dict

    @classmethod
    def load(cls, filename: str):
        return cls(pred_dict=load_pkl.load(filename))

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        return cls(pred_dict=pred_dict)

    def save(self, filename: str):
        save_pkl(filename, self.pred_dict)

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = self.pred_dict[dataset][fold][split_key]
            if models is None:
                models = model_results.keys()
            return np.array([model_results[model] for model in models])

        return [get_split(split, models) for split in splits]

    def get_dataset(self, dataset: str) -> DatasetPredictionsDict:
        return self.pred_dict[dataset]

    @property
    def models(self) -> List[str]:
        models = []
        for dataset in self.datasets:
            models.append(self.models_available_in_dataset(dataset))
        # returns models that appears in all lists, eg that are available for all datasets, folds and splits
        return sorted(set([x for l in models for x in l]))

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

    def _restrict_folds(self, folds: List[int]):
        folds_cur = self.folds
        valid_folds_set = set(folds)
        size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
        print(f'Restricting Folds... (Shrinking from {len(folds_cur)} -> {len(valid_folds_set)} folds)')
        print(f'OLD zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
        folds_to_remove = [f for f in folds_cur if f not in valid_folds_set]
        for f in folds_to_remove:
            self.remove_fold(f)
        size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
        print(f'NEW zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')

    @property
    def datasets(self) -> List[str]:
        return list(self.pred_dict.keys())

    def remove_fold(self, fold: int):
        for t in self.pred_dict.keys():
            self.pred_dict[t].pop(fold, None)

    def remove_dataset(self, dataset: str):
        self.pred_dict.pop(dataset)

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        self.pred_dict = {rename_dict.get(k, k): v for k, v in self.pred_dict.items()}

    @property
    def folds(self) -> List[int]:
        first = next(iter(self.pred_dict.values()))
        return sorted(list(first.keys()))


class TabularPicklePerTaskPredictions(TabularModelPredictions):
    # TODO: Consider saving/loading at the task level rather than the dataset level
    # TODO: Consider reducing the hackiness of rename_dict_inv, need to call `.get` in multiple places, makes code dupe
    def __init__(self, tasks_to_models: Dict[str, Dict[int, List[str]]], output_dir: str):
        """
        Stores on pickle per task and load data in a lazy fashion which allows to reduce significantly the memory
        footprint.
        :param tasks_to_models:
        :param output_dir:
        """
        self.tasks_to_models = tasks_to_models
        self.models_removed = set()
        self.output_dir = Path(output_dir)
        self.rename_dict_inv = {}
        assert self.output_dir.is_dir()
        for f in self.folds:
            assert isinstance(f, int)

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        dataset = self.rename_dict_inv.get(dataset, dataset)
        pred_dict = self._load_dataset(dataset)
        models_valid = self.models_available_in_dataset(dataset)
        if models is None:
            models = models_valid
        else:
            models_valid_set = set(models_valid)
            for m in models:
                assert m in models_valid_set, f"Model {m} is not valid for dataset {dataset} on fold {fold}! " \
                                              f"Valid models: {models_valid}"

        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = pred_dict[fold][split_key]
            return np.array([model_results[model] for model in models])

        available_model_mask = np.array([i for i, model in enumerate(models) if model not in self.models_removed])
        return [get_split(split, models)[available_model_mask] for split in splits]

    def models_available_in_dataset(self, dataset: str) -> List[str]:
        """Returns the models available on both validation and test splits on all tasks in a dataset"""
        models = []
        folds = self.folds
        dataset = self.rename_dict_inv.get(dataset, dataset)
        dataset_task_models = self.tasks_to_models[dataset]
        for fold in folds:
            if fold not in dataset_task_models:
                # if a fold is missing, no models has all folds.
                return []
        for fold in folds:
            task_models = dataset_task_models[fold]
            models.append(set(task_models))
        # returns models that appears in all lists, eg that are available for all folds and splits
        models = sorted(list(set.intersection(*map(set, models))))
        models = [m for m in models if m not in self.models_removed]
        return models

    def get_dataset(self, dataset: str) -> DatasetPredictionsDict:
        dataset = self.rename_dict_inv.get(dataset, dataset)
        return self._load_dataset(dataset=dataset)

    def _load_dataset(self, dataset: str, enforce_folds: bool = True) -> DatasetPredictionsDict:
        filename = str(self.output_dir / f'{dataset}.pkl')
        out = load_pkl.load(filename)
        if enforce_folds:
            valid_folds = set(self.tasks_to_models[dataset])
            folds = list(out.keys())
            for f in folds:
                if f not in valid_folds:
                    out.pop(f)
        return out

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        task_to_models = pred_proba.models_available_in_task_dict()
        print(f"saving .pkl files in folder {output_dir}")
        for dataset in tqdm(datasets):
            filename = str(output_dir / f'{dataset}.pkl')
            save_pkl(filename, pred_dict[dataset])
        cls._save_metadata(output_dir=output_dir, dataset_to_models=task_to_models)
        return cls(tasks_to_models=task_to_models, output_dir=output_dir)

    def save(self, output_dir: str):
        print(f"saving into {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata(output_dir, self.tasks_to_models)
        print(f"copy .pkl files from {self.output_dir} to {output_dir}")
        for file in self.output_dir.glob("*.pkl"):
            shutil.copyfile(file, output_dir / file.name)

    @classmethod
    def load(cls, filename: str):
        filename = Path(filename)
        metadata = cls._load_metadata(filename)
        dataset_to_models = metadata["dataset_to_models"]

        return cls(
            tasks_to_models=dataset_to_models,
            output_dir=filename,
        )

    @property
    def folds(self) -> List[int]:
        first = next(iter(self.tasks_to_models.values()))
        return sorted(list(first.keys()))

    @property
    def datasets(self):
        rename_dict_inv = {v: k for k, v in self.rename_dict_inv.items()}
        return [rename_dict_inv.get(d, d) for d in self.tasks_to_models.keys()]

    def _restrict_models(self, models: List[str]):
        for model in self.models:
            if model not in models:
                self.models_removed.add(model)

    def _restrict_folds(self, folds: List[int]):
        valid_folds_set = set(folds)
        for d in self.tasks_to_models:
            folds_in_dataset = list(self.tasks_to_models[d].keys())
            for f in folds_in_dataset:
                if f not in valid_folds_set:
                    self.tasks_to_models[d].pop(f)

    def remove_dataset(self, dataset: str):
        if dataset in self.datasets:
            self.tasks_to_models.pop(self.rename_dict_inv.get(dataset, dataset))

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        self.rename_dict_inv = {v: k for k, v in rename_dict.items()}

    @staticmethod
    def _save_metadata(output_dir, dataset_to_models):
        metadata = {
            "dataset_to_models": dataset_to_models,
        }
        save_pkl(path=str(output_dir / "metadata.pkl"), object=metadata)

    @staticmethod
    def _load_metadata(output_dir):
        return load_pkl.load(path=str(output_dir / "metadata.pkl"))

    @property
    def models(self) -> List[str]:
        res = set()
        for d in self.tasks_to_models.keys():
            for f in self.tasks_to_models[d].keys():
                for model in self.tasks_to_models[d][f]:
                    if model not in self.models_removed:
                        res.add(model)
        return sorted(list(res))


class TabularNpyPerTaskPredictions(TabularModelPredictions):
    def __init__(self, output_dir: str, dataset_shapes: Dict[str, Tuple[int, int, int]], models, folds):
        self._output_dir = output_dir
        self._dataset_shapes = dataset_shapes
        self._models = models
        self._folds = folds
        self.models_removed = set()

    def _predict_from_dataset(self, dataset: str) -> np.array:
        evals = np.load(Path(self._output_dir) / f"{dataset}.npy")
        num_val, num_test, output_dim = self._dataset_shapes[dataset]
        assert evals.shape[0] == num_val + num_test
        assert evals.shape[-1] == output_dim
        # (num_train/num_test, n_folds, n_models, output_dim)
        return evals[:num_val], evals[num_val:]

    def search_index(self, l, x):
        for i, y in enumerate(l):
            if y == x:
                return i

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        """
        :return: for each split, a tensor with shape (num_models, num_points) for regression and
        (num_models, num_points, num_classes) for classification corresponding the predictions of the model.
        """
        model_indices = {model: i for i, model in enumerate(self.models)}
        res = []
        # (num_train/num_test, n_folds, n_models, output_dim)
        val_evals, test_evals = self._predict_from_dataset(dataset)
        for split in splits:
            tensor = val_evals if split == "val" else test_evals
            if models is None:
                res.append(tensor[:, fold, :])
            else:
                res.append(tensor[:, fold, [model_indices[m] for m in models]])

        res = [np.swapaxes(x, 0, 1) for x in res]
        # squeeze last dim to be uniform with other part of the code
        res = [
            np.squeeze(x, axis=-1) if x.shape[-1] == 1 else x
            for x in res
        ]
        return res

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        models = pred_proba.models
        print(f"saving .pkl files in folder {output_dir}")
        dataset_shapes = {}
        for dataset in datasets:
            # (num_samples, n_folds, n_models, output_dim)
            val_evals, test_evals = cls._stack_pred(pred_dict[dataset], models)
            dataset_shapes[dataset] = (len(val_evals), len(test_evals), val_evals.shape[-1])
            evals = np.concatenate([val_evals, test_evals], axis=0)
            np.save(output_dir / f"{dataset}.npy", evals)
        cls._save_metadata(
            output_dir=output_dir,
            dataset_shapes=dataset_shapes,
            models=models,
            folds=pred_proba.folds,
        )

        return cls(
            dataset_shapes=dataset_shapes,
            output_dir=output_dir,
            models=models,
            folds=pred_proba.folds,
        )

    @staticmethod
    def _stack_pred(fold_dict: Dict[int, Dict[str, Dict[str, np.array]]], models):
        """
        :param fold_dict: dictionary mapping fold to split to config name to predictions
        :return:
        """
        # split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
        num_samples_val = min(len(config_evals) for config_evals in fold_dict[0]["pred_proba_dict_val"].values())
        num_samples_test = min(len(config_evals) for config_evals in fold_dict[0]["pred_proba_dict_test"].values())
        output_dims = set(
            config_evals.shape[1] if config_evals.ndim > 1 else 1
            for fold in fold_dict.values()
            for split in fold.values()
            for config_evals in split.values()
        )
        assert len(output_dims) == 1
        output_dim = next(iter(output_dims))
        n_folds = len(fold_dict)
        n_models = len(fold_dict[0]["pred_proba_dict_val"])
        val_res = np.zeros((num_samples_val, n_folds, n_models, output_dim))
        test_res = np.zeros((num_samples_test, n_folds, n_models, output_dim))
        def expand_if_scalar(x):
            return x if output_dim > 1 else np.expand_dims(x, axis=-1)

        for n_fold in range(n_folds):
            for n_model, model in enumerate(models):
                val_res[:, n_fold, n_model, :] = expand_if_scalar(
                    fold_dict[n_fold]["pred_proba_dict_val"][model][:num_samples_val]
                )
                test_res[:, n_fold, n_model, :] = expand_if_scalar(
                    fold_dict[n_fold]["pred_proba_dict_test"][model][:num_samples_test]
                )
        return val_res, test_res

    @staticmethod
    def _save_metadata(output_dir, dataset_shapes, models, folds):
        with open(output_dir / "metadata.json", "w") as f:
            metadata = {
                "dataset_shapes": dataset_shapes,
                "models": models,
                "folds": folds,
            }
            json.dump(metadata, f)

    @staticmethod
    def _load_metadata(output_dir):
        with open(output_dir / "metadata.json", "r") as f:
            return json.load(f)

    @property
    def datasets(self) -> List[str]:
        return list(self._dataset_shapes.keys())

    def models_available_in_dataset(self, dataset: str) -> List[str]:
        return [m for m in self._models if m not in self.models_removed]

    @property
    def folds(self) -> List[int]:
        return self._folds

    @property
    def models(self) -> List[int]:
        return [m for m in self._models if m not in self.models_removed]

    def _restrict_models(self, models: List[str]):
        for model in self.models:
            if model not in models:
                self.models_removed.add(model)

    def save(self, output_dir: str):
        print(f"saving into {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata(output_dir, dataset_shapes=self._dataset_shapes, models=self._models, folds=self._folds)
        print(f"copy .npy files from {self._output_dir} to {output_dir}")
        for file in self._output_dir.glob("*.npy"):
            shutil.copyfile(file, output_dir / file.name)

    @classmethod
    def load(cls, filename: str):
        filename = Path(filename)
        metadata = cls._load_metadata(filename)
        return cls(
            output_dir=filename,
            **metadata
        )
