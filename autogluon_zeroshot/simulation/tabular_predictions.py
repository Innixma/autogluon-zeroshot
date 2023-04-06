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

# dictionary mapping dataset to fold to split to config name to predictions
TabularPredictionsDict = Dict[str, Dict[int, Dict[str, Dict[str, np.array]]]]


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

    def models_available_in_dataset(self, dataset: str, fold: int) -> List[str]:
        """:returns the models available on both validation and test splits on all folds."""
        raise NotImplementedError()

    @property
    def models(self) -> List[str]:
        """
        :return: list of models present in at least one dataset.
        """
        raise NotImplementedError()

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
            assert d in self.datasets, f"cannot restrict {d} which is not in available datasets {self.datasets()}."
        self._restrict_datasets(datasets)

    def _restrict_datasets(self, datasets: List[str]):
        for dataset in self.datasets:
            if dataset not in datasets:
                self.remove_dataset(dataset)

    @property
    def datasets(self) -> List[str]:
        raise NotImplementedError()

    @property
    def folds(self) -> List[int]:
        raise NotImplementedError()

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

    def models_available_in_dataset(self, dataset: str) -> List[str]:
        models = []
        for fold in self.folds:
            if fold in self.pred_dict[dataset]:
                for split in ["pred_proba_dict_val", "pred_proba_dict_test"]:
                    models.append(set(self.pred_dict[dataset][fold][split].keys()))
            else:
                # if a fold is missing, no models has all folds.
                return []
        # returns models that appears in all lists, eg that are available for all folds and splits
        return sorted(list(set.intersection(*map(set, models))))

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

    @property
    def datasets(self) -> List[str]:
        return list(self.pred_dict.keys())

    def remove_dataset(self, dataset: str):
        self.pred_dict.pop(dataset)

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        self.pred_dict = {rename_dict.get(k, k): v for k, v in self.pred_dict.items()}

    @property
    def folds(self) -> List[int]:
        # todo we could assert that the same number of folds exists in all cases
        first = next(iter(self.pred_dict.values()))
        return list(first.keys())


class TabularPicklePerTaskPredictions(TabularModelPredictions):
    def __init__(self, dataset_to_models: Dict[str, List[str]], output_dir: str):
        """
        Stores on pickle per task and load data in a lazy fashion which allows to reduce significantly the memory
        footprint.
        :param dataset_to_models:
        :param output_dir:
        """
        self.dataset_to_models = dataset_to_models
        self.models_removed = set()
        self.output_dir = Path(output_dir)
        self.rename_dict_inv = {}
        assert self.output_dir.is_dir()

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        dataset = self.rename_dict_inv.get(dataset, dataset)
        pred_dict = self._load_dataset(dataset)
        if models is None:
            models = self.models_available_in_dataset(dataset)

        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = pred_dict[fold][split_key]
            return np.array([model_results[model] for model in models])

        available_model_mask = np.array([i for i, model in enumerate(models) if model not in self.models_removed])
        return [get_split(split, models)[available_model_mask] for split in splits]

    def _load_dataset(self, dataset: str) -> Dict:
        filename = str(self.output_dir / f'{dataset}.pkl')
        return load_pkl.load(filename)

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        dataset_to_models = {
            dataset: pred_proba.models_available_in_dataset(dataset)
            for dataset in datasets
        }
        print(f"saving .pkl files in folder {output_dir}")
        for dataset in tqdm(datasets):
            filename = str(output_dir / f'{dataset}.pkl')
            save_pkl(filename, pred_dict[dataset])
        cls._save_metadata(output_dir=output_dir, dataset_to_models=dataset_to_models)
        return cls(dataset_to_models=dataset_to_models, output_dir=output_dir)

    def save(self, output_dir: str):
        print(f"saving into {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata(output_dir, self.dataset_to_models)
        print(f"copy .pkl files from {self.output_dir} to {output_dir}")
        for file in self.output_dir.glob("*.pkl"):
            shutil.copyfile(file, output_dir / file.name)

    @classmethod
    def load(cls, filename: str):
        filename = Path(filename)
        metadata = cls._load_metadata(filename)
        dataset_to_models = metadata["dataset_to_models"]

        return cls(
            dataset_to_models=dataset_to_models,
            output_dir=filename,
        )

    def models_available_in_dataset(self, dataset: str) -> List[str]:
        # todo handle slice
        return self.dataset_to_models[self.rename_dict_inv.get(dataset, dataset)]

    @property
    def folds(self) -> List[int]:
        # TODO
        return list(range(10))

    @property
    def datasets(self):
        rename_dict_inv = {v: k for k, v in self.rename_dict_inv.items()}
        return [rename_dict_inv.get(d, d) for d in self.dataset_to_models.keys()]

    def _restrict_models(self, models: List[str]):
        for model in self.models:
            if model not in models:
                self.models_removed.add(model)

    def remove_dataset(self, dataset: str):
        if dataset in self.datasets:
            self.dataset_to_models.pop(self.rename_dict_inv.get(dataset, dataset))

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        self.rename_dict_inv = {v: k for k, v in rename_dict.items()}

    @staticmethod
    def _save_metadata(output_dir, dataset_to_models):
        with open(output_dir / "metadata.json", "w") as f:
            metadata = {
                "dataset_to_models": dataset_to_models,
            }
            json.dump(metadata, f)

    @staticmethod
    def _load_metadata(output_dir):
        with open(output_dir / "metadata.json", "r") as f:
            return json.load(f)

    @property
    def models(self) -> List[str]:
        res = set()
        for models in self.dataset_to_models.values():
            for model in models:
                if model not in self.models_removed:
                    res.add(model)
        return list(res)


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
