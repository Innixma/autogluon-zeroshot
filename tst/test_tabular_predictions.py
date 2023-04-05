import tempfile

from math import prod
from typing import List

import numpy as np
from pathlib import Path

import pytest

from autogluon_zeroshot.simulation.tabular_predictions import TabularPicklePredictions, TabularPredictionsDict, \
    TabularPicklePerTaskPredictions, TabularNpyPerTaskPredictions


def generate_dummy(shape, models):
    return {
        model: np.arange(prod(shape)).reshape(shape) + int(model)
        for model in models
    }


def generate_artificial_dict(
        num_folds: int,
        models: List[str],
        dataset_shapes={
            "d1": ((20,), (50,)),
            "d2": ((10,), (5,)),
            "d3": ((4, 3), (8, 3)),
        },
):
    # dictionary mapping dataset to fold to split to config name to predictions
    pred_dict: TabularPredictionsDict = {
        dataset: {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in range(num_folds)
        }
        for dataset, (val_shape, test_shape) in dataset_shapes.items()
    }
    return pred_dict


# def check_synthetic_data_pickle(cls=TabularPicklePredictions):
@pytest.mark.parametrize("cls", [TabularPicklePredictions, TabularPicklePerTaskPredictions, TabularNpyPerTaskPredictions])
def test_synthetic_data(cls):
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]

    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)

    with tempfile.TemporaryDirectory() as tmpdirname:

        # 1) construct pred proba from dictionary
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        assert set(pred_proba.models_available_in_dataset(dataset="d1")) == set(models)
        filename = str(Path(tmpdirname) / "dummy")

        # 2) save it and reload it
        pred_proba.save(filename)
        pred_proba = cls.load(filename)

        # 3) checks that output is as expected after serializing/deserializing
        assert pred_proba.datasets == list(dataset_shapes.keys())
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            print(dataset, val_shape, test_shape)
            val_score, test_score = pred_proba.predict(dataset=dataset, fold=2, models=models, splits=["val", "test"])
            assert val_score.shape == tuple([num_models] + list(val_shape))
            assert test_score.shape == tuple([num_models] + list(test_shape))
            for i, model in enumerate(models):
                assert np.allclose(val_score[i], generate_dummy(val_shape, models)[model])
                assert np.allclose(test_score[i], generate_dummy(test_shape, models)[model])


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePerTaskPredictions,
    # TabularNpyPerTaskPredictions
    # TODO restricting models with this format does not work which is ok as this
    #  format is not  currently used in experiments.
])
def test_restrict_models(cls):
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]
    num_sub_models = num_models // 2
    sub_models = models[:num_sub_models]
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        pred_proba.restrict_models(sub_models)
        assert sorted(pred_proba.models) == sorted(sub_models)

        # make sure shapes matches what is expected
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            print(dataset, val_shape, test_shape)
            val_score, test_score = pred_proba.predict(dataset=dataset, fold=2, models=sub_models, splits=["val", "test"])
            assert val_score.shape == tuple([num_sub_models] + list(val_shape))
            assert test_score.shape == tuple([num_sub_models] + list(test_shape))
            for i, model in enumerate(sub_models):
                assert np.allclose(val_score[i], generate_dummy(val_shape, sub_models)[model])
                assert np.allclose(test_score[i], generate_dummy(test_shape, sub_models)[model])


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePerTaskPredictions,
])
def test_restrict_datasets(cls):
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        pred_proba.restrict_datasets(["d1", "d3"])
        assert pred_proba.datasets == ["d1", "d3"]

        # make sure shapes matches what is expected
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            if dataset == "d2":
                continue
            print(dataset, val_shape, test_shape)
            val_score, test_score = pred_proba.predict(dataset=dataset, fold=2, models=models, splits=["val", "test"])
            assert val_score.shape == tuple([num_models] + list(val_shape))
            assert test_score.shape == tuple([num_models] + list(test_shape))
            for i, model in enumerate(models):
                assert np.allclose(val_score[i], generate_dummy(val_shape, models)[model])
                assert np.allclose(test_score[i], generate_dummy(test_shape, models)[model])

@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePerTaskPredictions,
])
def test_restrict_datasets_dense(cls):
    val_shape = (4, 3)
    test_shape = (8, 3)
    pred_dict = {
        "d1": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, ["1", "2", "3"]),
                "pred_proba_dict_test": generate_dummy(test_shape, ["1", "2", "3"]),
            }
            for fold in range(10)
        },
        "d2": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, ["2", "3"]),
                "pred_proba_dict_test": generate_dummy(test_shape, ["1", "3"]),
            }
            for fold in range(10)
        },
        "d3": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, ["1", "2", "3"]),
                "pred_proba_dict_test": generate_dummy(test_shape, ["1", "2", "3"]),
            }
            for fold in range(10)
        },
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)

        models = ["1", "2", "3"]
        valid_datasets = [
            dataset
            for dataset in pred_proba.datasets
            if all(m in pred_proba.models_available_in_dataset(dataset) for m in models)
        ]
        assert valid_datasets == ["d1", "d3"]
        pred_proba.restrict_datasets(valid_datasets)
        assert pred_proba.datasets == ["d1", "d3"]


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePerTaskPredictions,
])
def test_restrict_datasets_missing_fold(cls):
    val_shape = (4, 3)
    test_shape = (8, 3)
    models = ["1", "2", "3"]

    pred_dict = {
        "d1": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in range(10)
        },
        "d2": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in [x for x in range(10) if x != 3]
        },
        "d3": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in range(10)
        },
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        assert pred_proba.models_available_in_dataset("d1") == models
        assert pred_proba.models_available_in_dataset("d2") == []
        assert pred_proba.models_available_in_dataset("d3") == models
        valid_datasets = [
            dataset
            for dataset in pred_proba.datasets
            if all(m in pred_proba.models_available_in_dataset(dataset) for m in models)
        ]
        assert valid_datasets == ["d1", "d3"]
        pred_proba.restrict_datasets(valid_datasets)
        assert pred_proba.datasets == ["d1", "d3"]


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePerTaskPredictions,
])
def test_advanced(cls):
    """Tests a variety of advanced functionality"""
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)

        datasets_og = pred_proba.datasets
        pred_proba.restrict_datasets(datasets_og)
        assert datasets_og == pred_proba.datasets

        folds_og = pred_proba.folds
        pred_proba.restrict_folds(folds_og)
        assert folds_og == pred_proba.folds

        models_og = pred_proba.models
        pred_proba.restrict_models(models_og)
        assert models_og == pred_proba.models

        with pytest.raises(AssertionError):
            pred_proba.restrict_datasets(["unknown_dataset"])
        with pytest.raises(AssertionError):
            pred_proba.restrict_folds(["unknown_fold"])
        with pytest.raises(AssertionError):
            pred_proba.restrict_models(["unknown_model"])

        pred_proba.restrict_datasets(["d1", "d3"])
        assert pred_proba.datasets == ["d1", "d3"]

        pred_proba.restrict_folds([1, 2])
        assert pred_proba.folds == [1, 2]

        pred_proba.restrict_models(["3", "7", "11"])
        assert pred_proba.models == sorted(["3", "7", "11"])

        with pytest.raises(AssertionError):
            pred_proba.restrict_datasets(datasets_og)
        with pytest.raises(AssertionError):
            pred_proba.restrict_folds(folds_og)
        with pytest.raises(AssertionError):
            pred_proba.restrict_models(models_og)

        # make sure shapes matches what is expected
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            for fold in folds_og:
                for models in [
                    ["3"],  # valid
                    ["11", "7"],  # valid
                    ["11", "2"],  # invalid
                    ["2", "4"],  # invalid
                ]:
                    models_are_valid = False not in [m in pred_proba.models for m in models]
                    should_raise = dataset not in pred_proba.datasets or fold not in pred_proba.folds or not models_are_valid
                    print(dataset, fold, val_shape, test_shape, models, should_raise)
                    if should_raise:
                        with pytest.raises(Exception):
                            pred_proba.predict(dataset=dataset, fold=fold, models=models, splits=["val", "test"])
                    else:
                        val_score, test_score = pred_proba.predict(dataset=dataset, fold=fold, models=models, splits=["val", "test"])
                        assert val_score.shape == tuple([len(models)] + list(val_shape))
                        assert test_score.shape == tuple([len(models)] + list(test_shape))
                        for i, model in enumerate(models):
                            assert np.allclose(val_score[i], generate_dummy(val_shape, models)[model])
                            assert np.allclose(test_score[i], generate_dummy(test_shape, models)[model])
