import pandas as pd


class SingleBestConfigScorer:
    def __init__(self,
                 df_results_by_dataset_with_score_val: pd.DataFrame,
                 datasets: list = None,
                 score_col: str = 'rank',
                 score_val_col: str = 'score_val',
                 model_col: str = 'framework',
                 dataset_col: str = 'dataset'):
        self.score_col = score_col
        self.score_val_col = score_val_col
        self.model_col = model_col
        self.dataset_col = dataset_col
        if datasets is not None:
            df_results_by_dataset_with_score_val = df_results_by_dataset_with_score_val[
                df_results_by_dataset_with_score_val[dataset_col].isin(datasets)]
        self.df_results_by_dataset_with_score_val = df_results_by_dataset_with_score_val
        self.datasets = list(self.df_results_by_dataset_with_score_val[dataset_col].unique())
        self.df_pivot_val = self.df_results_by_dataset_with_score_val.pivot_table(index=self.model_col, columns=self.dataset_col, values=self.score_val_col)

    def get_configs_df(self, configs: list) -> pd.DataFrame:
        best_val_model_series = self.df_pivot_val.loc[configs].idxmax(axis=0).to_frame(name=self.model_col)
        best_val_model_by_dataset_df = self.df_results_by_dataset_with_score_val.merge(best_val_model_series, on=[self.dataset_col, self.model_col])
        return best_val_model_by_dataset_df

    def score(self, configs: list) -> float:
        best_val_model_by_dataset_df = self.get_configs_df(configs=configs)
        avg_error_real = best_val_model_by_dataset_df[self.score_col].mean()
        # this is the error without knowing the test score of each model and oracle picking the best,
        # instead using validation score to pick best
        return avg_error_real

    def subset(self, datasets):
        return self.__class__(
            df_results_by_dataset_with_score_val=self.df_results_by_dataset_with_score_val,
            datasets=datasets,
            score_col=self.score_col,
            score_val_col=self.score_val_col,
            model_col=self.model_col,
            dataset_col=self.dataset_col,
        )

