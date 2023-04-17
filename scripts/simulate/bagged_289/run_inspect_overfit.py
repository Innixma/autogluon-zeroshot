from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_fetcher import get_context
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation_debug


if __name__ == '__main__':
    # context_name = 'BAG_D244_F10_C608_FULL'
    context_name = 'BAG_D279_F10_C608_FULL'
    benchmark_context = get_context(context_name)
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load()
    zsc.print_info()

    # NOTE: For speed of simulation, it is recommended backend='ray'
    backend = 'ray'

    # configs = get_configs_small()
    configs = None

    datasets = zsc.get_dataset_folds()
    # datasets = zsc.get_dataset_folds(problem_type=['binary', 'multiclass'])

    config_scorer = SingleBestConfigScorer.from_zsc(
        zeroshot_simulator_context=zsc,
        datasets=datasets,
    )

    run_zs_simulation_debug(
        zsc=zsc,
        config_scorer=config_scorer,
        n_splits=5,
        n_repeats=2,
        configs=configs,
        config_generator_kwargs={
            'num_zeroshot': 20,
            'removal_stage': True,
        },
        backend=backend,
    )
