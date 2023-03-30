from . import intersect_folds_and_datasets


# TODO: Prune zeroshot_gt based on zeroshot_pred_proba final datasets
def load_zeroshot_input(path_pred_proba, path_gt, zsc, lazy_format: bool = False):
    zeroshot_gt = zsc.load_groundtruth(path_gt=path_gt)
    zeroshot_pred_proba = zsc.load_pred(
        pred_pkl_path=path_pred_proba,
        lazy_format=lazy_format,
    )

    # keep only dataset whose folds are all present
    intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt)
    models = zeroshot_pred_proba.models
    valid_datasets = [
        dataset
        for dataset in zeroshot_pred_proba.datasets
        if all(m in zeroshot_pred_proba.models_available_in_dataset(dataset) for m in models)
    ]
    if len(valid_datasets) < len(zeroshot_pred_proba.datasets):
        print(f"Restrict to {len(valid_datasets)} that contains all folds (from {len(zeroshot_pred_proba.datasets)}).")
        zeroshot_pred_proba.restrict_datasets(datasets=valid_datasets)

    zsc.subset_models(zeroshot_pred_proba.models)
    zsc.subset_datasets(zeroshot_pred_proba.datasets)
    zeroshot_pred_proba.restrict_models(zsc.get_configs())
    zeroshot_gt = prune_zeroshot_gt(zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)

    return zeroshot_pred_proba, zeroshot_gt, zsc


def prune_zeroshot_gt(zeroshot_pred_proba, zeroshot_gt):
    num_datasets_start = len(zeroshot_gt)
    datasets = set(zeroshot_pred_proba.datasets)
    datasets_gt = list(zeroshot_gt.keys())
    for d in datasets_gt:
        if d not in datasets:
            zeroshot_gt.pop(d)
    num_datasets_end = len(zeroshot_gt)
    print(f'Aligning GT with pred_proba... (Dataset count {num_datasets_start} -> {num_datasets_end})')
    assert len(datasets) == num_datasets_end
    return zeroshot_gt
