import ast
import copy
from typing import List

import numpy as np
from dataclasses import dataclass

import pandas as pd
import ray

from syne_tune.experiments import load_experiments_df
from tqdm import tqdm

from autogluon_zeroshot.portfolio.zeroshot_selection import zeroshot_configs
from autogluon_zeroshot.simulation.repository import EvaluationRepository


@dataclass
class ResultRow:
    dataset: str
    fold: int
    method: str
    test_error: float
    rank: float
    normalized_score: float
    config_selected: list = None


def automl_results(repo: EvaluationRepository, dataset_names: List[str], n_folds: int, rank_scorer, normalized_scorer) -> List[ResultRow]:
    """
    :return: evaluation of AutoGluon medium/high/best quality.
    """
    automl_df = copy.deepcopy(repo._zeroshot_context.df_results_by_dataset_automl)
    automl_df['fold'] = automl_df['dataset'].map(repo._zeroshot_context.dataset_name_to_fold_dict)
    automl_df['tid'] = automl_df['dataset'].map(repo._zeroshot_context.dataset_name_to_tid_dict)
    automl_df['task'] = automl_df['dataset']
    # FIXME: Instead of returning list of "ResultRow", return this dataframe
    automl_df['dataset'] = automl_df['tid'].map(repo._zeroshot_context.tid_to_dataset_dict)

    rows_automl = []
    for dataset in tqdm(dataset_names):
        for fold in range(n_folds):
            dataset_fold_name = f"{repo.dataset_to_taskid(dataset)}_{fold}"
            automl_df_fold = automl_df[automl_df['task'] == dataset_fold_name]
            task_automl_dict = automl_df_fold.T.to_dict()

            for k, v in task_automl_dict.items():
                metric_error = v['metric_error']
                rows_automl.append(ResultRow(
                    dataset=v['dataset'],
                    fold=v['fold'],
                    method=v['framework'],
                    test_error=metric_error,
                    rank=rank_scorer.rank(dataset_fold_name, metric_error),
                    normalized_score=normalized_scorer.rank(dataset_fold_name, metric_error),
                ))

    return rows_automl


@ray.remote
def _zeroshot_results_ray(i,
                          repo: EvaluationRepository,
                          df_rank: pd.DataFrame,
                          train_dataset_names: List[str],  # FIXME: Use this instead of assuming LOO
                          test_dataset_names: List[str],  # FIXME: Allow more than 1 element
                          n_folds: int,  # TODO: Switch to list of folds to have more control
                          rank_scorer,
                          normalized_scorer,
                          ensemble_size: int,
                          portfolio_size) -> List[ResultRow]:
    rows_zeroshot = []

    assert len(test_dataset_names) == 1
    dataset = test_dataset_names[0]

    taskid = repo.dataset_to_taskid(dataset)
    train_datasets = [x for x in df_rank.columns if x.split("_")[0] != str(taskid)]
    # assert train_datasets == train_dataset_names

    indices = zeroshot_configs(-df_rank[train_datasets].values.T, portfolio_size)
    portfolio_configs = [df_rank.index[i] for i in indices]

    # run best base model and ensemble
    suffix = f"-{portfolio_size}-{ensemble_size}"
    if ensemble_size:
        suffix += " (ensemble)"
    test_errors = repo.evaluate_ensemble(
        dataset_names=[dataset],
        config_names=portfolio_configs,
        ensemble_size=ensemble_size,
        rank=False,
        backend='native',
    )
    assert test_errors.shape[0] == 1  # we send one model, we should get one row back
    for fold in range(n_folds):
        test_error = test_errors[0][fold]
        dataset_fold_name = f"{repo.dataset_to_taskid(dataset)}_{fold}"
        rows_zeroshot.append(ResultRow(
            dataset=dataset,
            fold=fold,
            method=f"Zeroshot{suffix}",
            test_error=test_error,
            rank=rank_scorer.rank(dataset_fold_name, test_error),
            normalized_score=normalized_scorer.rank(dataset_fold_name, test_error),
            config_selected=portfolio_configs,
        ))
    print(f'Ray job finished {i}:\t{dataset}')
    return rows_zeroshot


def zeroshot_results_ray(
    repo: EvaluationRepository, dataset_names: List[str], n_folds: int, rank_scorer, normalized_scorer,
    ensemble_sizes: List[int] =[1, 20], portfolio_sizes: List[int]=[5, 10, 20, 40, 80]
):
    dd = repo._zeroshot_context.df_results_by_dataset_vs_automl
    df_rank = dd.pivot_table(index="framework", columns="dataset", values="score_val").rank()
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))

    if not ray.is_initialized():
        ray.init()

    repo_ray = ray.put(repo)
    df_rank_ray = ray.put(df_rank)
    rank_scorer_ray = ray.put(rank_scorer)
    normalized_scorer_ray = ray.put(normalized_scorer)

    results = []
    i = 1
    for dataset in dataset_names:
        for portfolio_size in portfolio_sizes:
            for ensemble_size in ensemble_sizes:
                # Create and execute all tasks in parallel
                results.append(_zeroshot_results_ray.remote(
                    i,
                    repo_ray,
                    df_rank_ray,
                    None,
                    [dataset],
                    n_folds,
                    rank_scorer_ray,
                    normalized_scorer_ray,
                    ensemble_size,
                    portfolio_size,
                ))
                i += 1
    print(f'Total ray jobs: {i}')
    rows_zeroshot: list = ray.get(results)
    rows_zeroshot = [item for sublist in rows_zeroshot for item in sublist]
    return rows_zeroshot


# TODO: Consider removing in favor of the ray-variant
def zeroshot_results(
        repo: EvaluationRepository, dataset_names: List[str], n_folds: int, rank_scorer, normalized_scorer,
        ensemble_sizes: List[int] =[1, 20], portfolio_sizes: List[int]=[5, 10, 20, 40, 80]
) -> List[ResultRow]:
    """
    :param ensemble_sizes: number of caruana sizes to consider
    :param portfolio_sizes: number of portfolio to consider
    :return: evaluation obtained on all combinations from `ensemble_sizes` and `portfolio_sizes`
    """
    dd = repo._zeroshot_context.df_results_by_dataset_vs_automl
    df_rank = dd.pivot_table(index="framework", columns="dataset", values="score_val").rank()
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))
    rows_zeroshot = []
    for dataset in tqdm(dataset_names):
        for portfolio_size in portfolio_sizes:
            for ensemble_size in ensemble_sizes:
                taskid = repo.dataset_to_taskid(dataset)
                train_datasets = [x for x in df_rank.columns if x.split("_")[0] != str(taskid)]
                indices = zeroshot_configs(-df_rank[train_datasets].values.T, portfolio_size)
                portfolio_configs = [df_rank.index[i] for i in indices]

                # run best base model and ensemble
                suffix = f"-{portfolio_size}-{ensemble_size}"
                if ensemble_size:
                    suffix += " (ensemble)"
                test_errors = repo.evaluate_ensemble(
                    dataset_names=[dataset],
                    config_names=portfolio_configs,
                    ensemble_size=ensemble_size,
                    rank=False,
                )
                assert test_errors.shape[0] == 1  # we send one model, we should get one row back
                for fold in range(n_folds):
                    test_error = test_errors[0][fold]
                    dataset_fold_name = f"{repo.dataset_to_taskid(dataset)}_{fold}"
                    rows_zeroshot.append(ResultRow(
                        dataset=dataset,
                        fold=fold,
                        method=f"Zeroshot{suffix}",
                        test_error=test_error,
                        rank=rank_scorer.rank(dataset_fold_name, test_error),
                        normalized_score=normalized_scorer.rank(dataset_fold_name, test_error),
                        config_selected=portfolio_configs,
                    ))
    return rows_zeroshot


def evaluate_tuning(
        repo: EvaluationRepository, dataset_names: List[str], n_folds: int, rank_scorer, normalized_scorer,
        expname="02-05-v2",
) -> List[ResultRow]:
    """
    :param expname: name of the experiment tag passed when launching `scripts/method_comparison/run_method_comparison.py`
    :return: results of top configurations found by local search and zeroshot when using LocalSearch, Zeroshot, Zeroshot ensemble.
    """
    def load_df_tuning(expname):
        name_filter = lambda path: expname in str(path)
        df_results = load_experiments_df(path_filter=name_filter)
        df_results["fold"] = df_results.apply(lambda row: int(row['tuner_name'].split("fold-")[1].split("-")[0]),
                                              axis=1)
        for col in ['configs', 'train_datasets', 'test_datasets']:
            df_results[col] = df_results[f'config_{col}'].apply(lambda x: ast.literal_eval(x))
        return df_results

    def extract_configs_from_tuning_results(df_results):
        rows = []
        for fold in sorted(df_results.fold.unique()):
            df_sub = df_results[(df_results.fold == fold) & (df_results.searcher == "localsearch")]
            # zeroshot config is always the first trial
            row_zeroshot = df_sub.loc[
                df_sub.trial_id == 0, ['configs', 'train_datasets', 'test_datasets', 'train_error', 'test_error']].head(
                1)
            # localsearch config is the one with lowest train error
            row_localsearch = df_sub.sort_values("train_error").loc[:,
                              ['configs', 'train_datasets', 'test_datasets', 'train_error', 'test_error']].head(1)
            assert row_localsearch["train_datasets"].values[0] == row_zeroshot["train_datasets"].values[0]
            rows.append({
                "metafold": fold,
                "zeroshot": row_zeroshot["configs"].values[0],
                "localsearch": row_localsearch["configs"].values[0],
                "train_datasets": row_localsearch["train_datasets"].values[0],
                "test_datasets": row_localsearch["test_datasets"].values[0],
            })
        return rows

    def taskid_to_config(tuning_rows, taskid):
        contains_task = lambda tasks: any(task.split("_")[0] == str(taskid) for task in tasks)
        matches = [row for row in tuning_rows if not contains_task(row['train_datasets'])]
        assert len(matches) >= 1
        return matches[0]

    df_results = load_df_tuning(expname=expname)
    if len(df_results) == 0:
        print(f"No tuning result could be found from the experiment tag passed {expname}, please run run_method_comparison.py with this tag first.")
        return

    tuning_rows = extract_configs_from_tuning_results(df_results)

    rows = []
    for dataset in tqdm(dataset_names):
        for suffix, ensemble_size in [("", 1), (f" (ensemble)", 20)]:
            for method in ["zeroshot", "localsearch"]:
                test_errors = repo.evaluate_ensemble(
                    dataset_names=[dataset],
                    config_names=taskid_to_config(tuning_rows, repo.dataset_to_taskid(dataset))[method],
                    ensemble_size=ensemble_size,
                    rank=False,
                )
                assert test_errors.shape[0] == 1  # we send one model, we should get one row back
                for fold in range(n_folds):
                    test_error = test_errors[0][fold]
                    dataset_fold_name = f"{repo.dataset_to_taskid(dataset)}_{fold}"
                    rows.append(ResultRow(
                        dataset=dataset,
                        fold=fold,
                        method=f"{method}{suffix}".capitalize(),
                        test_error=test_error,
                        rank=rank_scorer.rank(dataset_fold_name, test_error),
                        normalized_score=normalized_scorer.rank(dataset_fold_name, test_error),
                    ))
    return rows
