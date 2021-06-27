from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFpr, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


DUMMY_COLS = ["JOB_TYPE", "DIAMETER_MM", "MATERIAL"]
NON_DUMMY_COLS = ["DEPTH_M", "LENGTH_M"]


def run_model(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[BaseEstimator, object, pd.DataFrame]:
    """
    Runs cross validation pipeline of all model variants and generates summary
    report on error metrics and feature importances.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.DataFrame

    Returns
    -------
    best_estimator : sklearn BaseEstimator
        Estimator from sklearn pipeline.
    summary_report : pandas Styler object
        Summary report styled for UI.
    feature_importances : pd.DataFrame
        Dataframe of feature importances or coefficients in model.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(), DUMMY_COLS),
            ("numericals", "passthrough", NON_DUMMY_COLS),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection", SelectFpr(f_regression, alpha=0.05)),
            ("clf", AdaBoostRegressor(DecisionTreeRegressor())),
        ]
    )
    parameters = [
        {
            "clf": [
                TransformedTargetRegressor(
                    regressor=LinearRegression(), func=np.log, inverse_func=np.exp
                )
            ],
        },
        {
            "clf": [AdaBoostRegressor(DecisionTreeRegressor())],
            "clf__base_estimator__max_depth": [5, 10],
            "clf__base_estimator__ccp_alpha": [0.1, 1, 10],
            "clf__learning_rate": [0.1, 1, 10],
        },
    ]

    grid_search = GridSearchCV(
        pipeline,
        parameters,
        n_jobs=-1,
        return_train_score=True,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        refit="rmse",
    )
    grid_search.fit(X, y)

    results = grid_search.cv_results_
    result_cols = ["Linear Regression"] + ["AdaBoost"] * 18
    report = pd.DataFrame(
        zip(
            result_cols,
            results["params"],
            results["mean_test_rmse"],
            results["mean_test_r2"],
            results["mean_train_rmse"],
            results["mean_train_r2"],
        ),
        columns=[
            "Model Name",
            "Params",
            "Mean Validation RMSE(£)",
            "Mean Validation R2(%)",
            "Mean Training RMSE(£)",
            "Mean Training R2(%)",
        ],
    ).sort_values(by="Mean Validation RMSE(£)", ascending=False)
    formatting = dict(
        zip(report.columns[2:], ["{:,.0f}", "{:.2%}", "{:,.0f}", "{:.2%}"])
    )

    best_estimator = grid_search.best_estimator_
    if type(best_estimator["clf"]) == AdaBoostRegressor:
        coeffs = best_estimator["clf"].feature_importances_
    elif type(best_estimator["clf"]) == TransformedTargetRegressor:
        coeffs = best_estimator["clf"].regressor_.coef_
    else:
        raise ValueError("Estimator type not supported")

    feature_names = (
        list(
            best_estimator.named_steps["preprocessor"]
            .transformers_[0][1]
            .get_feature_names(DUMMY_COLS)
        )
        + NON_DUMMY_COLS
    )
    feature_importances = pd.DataFrame(
        coeffs,
        index=np.array(feature_names)[
            best_estimator.named_steps["feature_selection"].get_support()
        ],
        columns=["Absolute Feature Importances"],
    )

    return (
        grid_search.best_estimator_,
        report.style.format(formatting),
        feature_importances,
    )
