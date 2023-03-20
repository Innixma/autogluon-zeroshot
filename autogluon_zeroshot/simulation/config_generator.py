import copy
from collections import defaultdict
import math
import random
import time
from typing import Any, Dict, List

import numpy as np
import ray
from sklearn.model_selection import RepeatedKFold

from .configuration_list_scorer import ConfigurationListScorer
from .simulation_context import ZeroshotSimulatorContext
from ..portfolio import Portfolio, PortfolioCV


@ray.remote
def score_config_ray(config_scorer, existing_configs, new_config) -> float:
    configs = existing_configs + [new_config]
    score = config_scorer.score(configs)
    return score


class ZeroshotConfigGenerator:
    def __init__(self, config_scorer, configs: List[str], backend='ray'):
        self.config_scorer = config_scorer
        self.all_configs = configs
        self.backend = backend

    def select_zeroshot_configs(self,
                                num_zeroshot: int,
                                zeroshot_configs: List[str] = None,
                                removal_stage=False,
                                removal_threshold=0,
                                config_scorer_test=None,
                                return_all_metadata: bool = False,
                                ) -> (List[Dict[str, Any]]):
        zeroshot_configs = [] if zeroshot_configs is None else copy.deepcopy(zeroshot_configs)
        metadata_list = []

        iteration = 0
        if self.backend == 'ray':
            if not ray.is_initialized():
                ray.init()
            config_scorer = ray.put(self.config_scorer)
            selector = self._select_ray
        else:
            config_scorer = self.config_scorer
            selector = self._select_sequential
        while len(zeroshot_configs) < num_zeroshot:
            iteration += 1
            # greedily search the config that would yield the lowest average rank if we were to evaluate it in combination
            # with previously chosen configs.

            valid_configs = [c for c in self.all_configs if c not in zeroshot_configs]
            if not valid_configs:
                break

            time_start = time.time()
            best_next_config, train_score_best = selector(valid_configs, zeroshot_configs, config_scorer)
            time_end = time.time()

            zeroshot_configs.append(best_next_config)
            fit_time = time_end - time_start
            msg = f'{iteration}\t: Train: {round(train_score_best, 2)}'
            if config_scorer_test:
                test_score = config_scorer_test.score(zeroshot_configs)
                msg += f'\t| Test: {round(test_score, 2)} \t| Overfit: {round(test_score-train_score_best, 2)}'
            else:
                test_score = None
            msg += f' | {round(fit_time, 2)}s | {self.backend} | {best_next_config}'
            # print('here, make metadata')
            metadata_out = dict(
                configs=copy.deepcopy(zeroshot_configs),
                new_config=best_next_config,
                train_score=train_score_best,
                test_score=test_score,
                num_configs=len(zeroshot_configs),
                fit_time=fit_time,
                backend=self.backend,
            )
            is_last = len(zeroshot_configs) >= num_zeroshot
            if return_all_metadata or is_last:
                metadata_list.append(metadata_out)

            print(msg)
        if removal_stage:
            raise NotImplementedError('Currently removal_stage=True does not work correctly')
            zeroshot_configs = self.prune_zeroshot_configs(zeroshot_configs, removal_threshold=removal_threshold)
        print(f"selected {zeroshot_configs}")
        # TODO: metadata_list not updated by prune_zeroshot_configs
        return metadata_list

    @staticmethod
    def _select_sequential(configs: list, prior_configs: list, config_scorer):
        best_next_config = None
        # todo could use np.inf but would need unit-test (also to check that ray/sequential returns the same selection)
        best_score = 999999999
        for config in configs:
            config_selected = prior_configs + [config]
            config_score = config_scorer.score(config_selected)
            if config_score < best_score:
                best_score = config_score
                best_next_config = config
        return best_next_config, best_score

    @staticmethod
    def _select_ray(configs: list, prior_configs: list, config_scorer):
        # Create and execute all tasks in parallel
        results = []
        for i in range(len(configs)):
            results.append(score_config_ray.remote(
                config_scorer,
                prior_configs,
                configs[i],
            ))
        result = ray.get(results)
        result_idx_min = result.index(min(result))
        best_next_config = configs[result_idx_min]
        best_score = result[result_idx_min]
        return best_next_config, best_score

    def prune_zeroshot_configs(self, zeroshot_configs: List[str], removal_threshold=0) -> List[str]:
        zeroshot_configs = copy.deepcopy(zeroshot_configs)
        best_score = self.config_scorer.score(zeroshot_configs)
        finished_removal = False
        while not finished_removal:
            best_remove_config = None
            for config in zeroshot_configs:
                config_selected = [c for c in zeroshot_configs if c != config]
                config_score = self.config_scorer.score(config_selected)

                if best_remove_config is None:
                    if config_score <= (best_score + removal_threshold):
                        best_score = config_score
                        best_remove_config = config
                else:
                    if config_score <= best_score:
                        best_score = config_score
                        best_remove_config = config
            if best_remove_config is not None:
                print(f'REMOVING: {best_score} | {best_remove_config}')
                zeroshot_configs.remove(best_remove_config)
            else:
                finished_removal = True
        return zeroshot_configs


class ZeroshotConfigGeneratorCV:
    def __init__(self,
                 n_splits: int,
                 zeroshot_simulator_context: ZeroshotSimulatorContext,
                 config_scorer: ConfigurationListScorer,
                 config_generator_kwargs=None,
                 configs: List[str] = None,
                 n_repeats: int = 1,
                 backend='ray'):
        """
        Runs zero-shot selection on `n_splits` ("train", "test") folds of datasets.
        For each split, zero-shot configurations are selected using the datasets belonging on the "train" split and the
        performance of the zero-shot configuration is evaluated using the datasets in the "test" split.
        :param n_splits: number of splits for RepeatedKFold
        :param zeroshot_simulator_context:
        :param config_scorer:
        :param configs:
        :param n_repeats: number of repeats for RepeatedKFold
        :param backend:
        """
        assert n_splits >= 2
        assert n_repeats >= 1
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        if config_generator_kwargs is None:
            config_generator_kwargs = {}
        self.config_generator_kwargs = config_generator_kwargs
        self.backend = backend
        self.config_scorer = config_scorer
        self.unique_datasets_fold = np.array(config_scorer.datasets)
        self.unique_datasets_map = zeroshot_simulator_context.dataset_name_to_tid_dict
        self.unique_datasets = set()
        self.dataset_parent_to_fold_map = dict()
        for d in self.unique_datasets_fold:
            dataset_parent = self.unique_datasets_map[d]
            self.unique_datasets.add(dataset_parent)
            if dataset_parent in self.dataset_parent_to_fold_map:
                self.dataset_parent_to_fold_map[dataset_parent].append(d)
            else:
                self.dataset_parent_to_fold_map[dataset_parent] = [d]
        for d in self.dataset_parent_to_fold_map:
            self.dataset_parent_to_fold_map[d] = sorted(self.dataset_parent_to_fold_map[d])
        self.unique_datasets = np.array((sorted(list(self.unique_datasets))))

        if configs is None:
            configs = zeroshot_simulator_context.get_configs()
        self.configs = configs

        self.kf = RepeatedKFold(n_splits=self.n_splits, random_state=0, n_repeats=self.n_repeats)

    def _get_dataset_parent_to_fold(self, dataset: str, num_folds=None) -> List[str]:
        if num_folds is None:
            return self.dataset_parent_to_fold_map[dataset]
        else:
            return self.dataset_parent_to_fold_map[dataset][:num_folds]

    def run_and_return_all_steps(self,
                                 sample_train_folds: int = None,
                                 sample_train_ratio: float = None,
                                 score_all: bool = True,
                                 score_final: bool = True,
                                 return_all_metadata: bool = True) -> List[PortfolioCV]:
        """
        Run cross-validated zeroshot simulation.

        :param sample_train_folds:
            Number of folds to filter training data to for each fold. Used for debugging.
            Lower values should result in worse test scores and higher overfit scores
            If set to a value larger than the folds available in the datasets, it will have no effect.
        :param sample_train_ratio:
            Ratio of datasets to filter training data to for each fold. Used for debugging.
            Lower values should result in worse test scores and higher overfit scores
        :param score_all: If True, calculates test score at each step of the zeroshot simulation process.
        :param score_final: If True, calculates test score for the final step of the zeroshot simulation process.
        :param return_all_metadata:
            If True, returns N elements in a list, with the index referring to the order of selection.
                Note: If folds differ in number of simulation steps, this will raise an exception.
                For example, config pruning as a post-processing step to greedy selection
                of N elements could have differing step counts.
            If False, returns a list with only 1 element corresponding to the final zeroshot config.
        """
        results_dict_by_len = defaultdict(list)
        for i, (train_index, test_index) in enumerate(self.kf.split(self.unique_datasets)):
            X_train, X_test = list(self.unique_datasets[train_index]), list(self.unique_datasets[test_index])
            len_X_train_og = len(X_train)
            len_X_test_og = len(X_test)
            if sample_train_ratio is not None and sample_train_ratio < 1:
                random.seed(0)
                num_samples = math.ceil(len(X_train) * sample_train_ratio)
                X_train = random.sample(X_train, num_samples)
            X_train_fold = []
            X_test_fold = []
            for d in X_train:
                X_train_fold += self._get_dataset_parent_to_fold(dataset=d, num_folds=sample_train_folds)
            for d in X_test:
                X_test_fold += self.dataset_parent_to_fold_map[d]
            len_X_train_fold = len(X_train_fold)
            len_X_train = len(X_train)
            print(f'Fitting Fold {i + 1}/{self.n_splits*self.n_repeats}... '
                  f'(n_splits={self.n_splits}, n_repeats={self.n_repeats})\n'
                  f'\tsample_train_folds={sample_train_folds} | sample_train_ratio={sample_train_ratio}\n'
                  f'\ttrain_datasets: {len_X_train}/{len_X_train_og} | train_tasks: {len_X_train_fold}\n'
                  f'\ttest_datasets : {len(X_test)}/{len_X_test_og} | test_tasks : {len(X_test_fold)}'
                  )
            metadata_fold = self.run_fold(X_train_fold,
                                          X_test_fold,
                                          score_all=score_all,
                                          score_final=score_final,
                                          return_all_metadata=return_all_metadata)
            for j, m in enumerate(metadata_fold):
                # FIXME: It is possible not all folds will have results match up correctly
                #  if we introduce config pruning.
                #  This logic should probably only be present in scenarios where we are debugging
                #  Otherwise we should only take the final result of each fold.
                results_fold_i = Portfolio(
                    configs=m['configs'],
                    train_score=m['train_score'],
                    test_score=m['test_score'],
                    train_datasets=X_train,
                    test_datasets=X_test,
                    train_datasets_fold=X_train_fold,
                    test_datasets_fold=X_test_fold,
                    fold=i + 1,
                )
                results_dict_by_len[j].append(results_fold_i)
        for val in results_dict_by_len.values():
            assert (self.n_splits * self.n_repeats) == len(val)  # Ensure no bugs such as only getting a subset of fold results
        portfolio_cv_list = [PortfolioCV(portfolios=v) for k, v in results_dict_by_len.items()]
        return portfolio_cv_list

    def run(self,
            sample_train_folds=None,
            sample_train_ratio=None,
            score_all=False,
            score_final=True) -> PortfolioCV:
        """
        Identical to `run_and_return_all_steps`, but the output is simply the final PortfolioCV.

        score_all is also set to False by default to speed up the simulation.
        """
        results_cv_list = self.run_and_return_all_steps(sample_train_folds=sample_train_folds,
                                                        sample_train_ratio=sample_train_ratio,
                                                        score_all=score_all,
                                                        score_final=score_final,
                                                        return_all_metadata=False)

        assert len(results_cv_list) == 1
        results_cv = results_cv_list[0]

        return results_cv

    def run_fold(self,
                 train_tasks: List[str],
                 test_tasks: List[str],
                 score_all=False,
                 score_final=True,
                 return_all_metadata=False) -> List[Dict[str, Any]]:
        config_scorer_train = self.config_scorer.subset(datasets=train_tasks)
        config_scorer_test = self.config_scorer.subset(datasets=test_tasks)

        zs_config_generator = ZeroshotConfigGenerator(config_scorer=config_scorer_train,
                                                      configs=self.configs,
                                                      backend=self.backend)

        num_zeroshot = self.config_generator_kwargs.get('num_zeroshot', 10)
        removal_stage = self.config_generator_kwargs.get('removal_stage', False)

        metadata_list = zs_config_generator.select_zeroshot_configs(
            num_zeroshot=num_zeroshot,
            removal_stage=removal_stage,
            config_scorer_test=config_scorer_test if score_all else None,
            return_all_metadata=return_all_metadata,
        )
        # deleting
        # FIXME: SPEEDUP WITH RAY
        # zeroshot_configs = zs_config_generator.prune_zeroshot_configs(zeroshot_configs, removal_threshold=0)

        if score_final and metadata_list[-1]['test_score'] is None:
            score = config_scorer_test.score(metadata_list[-1]['configs'])
            print(f'test_score: {score}')
            metadata_list[-1]['test_score'] = score

        return metadata_list
