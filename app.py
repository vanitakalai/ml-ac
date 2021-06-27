import os
from typing import Tuple, Optional
import pickle
import pandas as pd
import numpy as np
import datetime
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from ml_ac.preprocessing import Preprocess, X_COLS
from ml_ac.model import run_model, DUMMY_COLS

sns.set_theme(style="darkgrid")


def setup_inputs() -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]
]:
    st.sidebar.subheader("Upload training data here:")
    file = st.sidebar.file_uploader("", type=".csv")

    st.sidebar.subheader("Set outlier threshold:")
    max_std_outlier = st.sidebar.slider("", min_value=1, max_value=5, value=3)

    X, y, rows = None, None, None
    if file is not None:
        p = Preprocess(max_std_outlier)
        X, y = p.load_data(file)
        X, y = p.remove_outliers(X, y)
        rows = p.removed_data(X)

    return X, y, rows


def plot_counts(dataset: pd.DataFrame, rows: pd.DataFrame) -> None:
    st.subheader(f"Total Number of samples : {len(dataset)}")
    expander = st.beta_expander(f"See {len(rows)} rows removed from dataset:")
    expander.table(rows)

    st.subheader("What kind of work are we receiving?")
    var = st.selectbox("Variable Name", dataset.columns)

    fig = sns.displot(x=dataset[var], aspect=3)
    st.pyplot(fig)


def plot_relationships(dataset: pd.DataFrame, y_name: str) -> None:
    st.subheader("What is the relationship to the cost?")
    var = st.selectbox("Variable Name:", dataset.columns)

    if var in DUMMY_COLS:
        fig = sns.displot(data=dataset, x=y_name, hue=var, kind="kde", aspect=3)
    else:
        fig = sns.FacetGrid(dataset, aspect=3)
        fig.map(sns.regplot, var, y_name)

    st.pyplot(fig)


@st.cache(show_spinner=False, allow_output_mutation=True)
def show_model(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    with st.spinner("Training model..."):
        estimator, report, feature_imps = run_model(X, y[y.columns[0]])

    date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(
        f"models/model_{date}.pkl",
        "wb",
    ) as f:
        pickle.dump(estimator, f)

    report.data["Date"] = date
    if os.path.exists("model_metrics.csv"):
        report.data.iloc[[0]].to_csv("model_metrics.csv", mode="a", header=False)
    else:
        report.data.iloc[[0]].to_csv("model_metrics.csv", mode="a", header=True)

    return estimator, report, feature_imps


def get_existing_model() -> Optional[object]:
    models = pd.Series(os.listdir("models"))
    if len(models) > 0:
        latest_model = (
            pd.to_datetime(models.str.split("_").str[-1].str.split(".").str[0])
            .sort_values(ascending=False)
            .iloc[0]
        )
        with open(
            f"models/model_{latest_model.strftime('%d-%m-%Y %H:%M:%S')}.pkl", "rb"
        ) as f:
            model = pickle.load(f)

        return model
    else:
        return None


def user_test_data(model: object) -> None:
    st.subheader("Estimate price for current job: ")
    cols = st.beta_columns((1, 1, 1, 1, 1))
    job_type = cols[0].selectbox("Job Type", ["Repair", "Replace"])
    material = cols[1].selectbox("Material", ["ST", "AC", "CI"])
    diameter = cols[2].selectbox("Diameter(mm)", [50, 63, 90, 125, 200, 250, 315, 450])
    depth = cols[3].number_input("Depth (m)", min_value=0.0, max_value=5.0, value=3.0)
    length = cols[4].number_input(
        "Length (m)", min_value=0.0, max_value=50.0, value=20.0
    )

    input_data = pd.DataFrame(
        [job_type, depth, diameter, material, length], index=X_COLS
    ).T
    y_pred = round(model.predict(input_data)[0], 2)
    st.subheader(f"Predicted cost for job = £{y_pred}")


def load_error_history() -> None:
    st.subheader("Tracking error metrics")
    errors = pd.read_csv("model_metrics.csv")[
        ["Date", "Mean Training RMSE(£)", "Mean Training R2(%)"]
    ]

    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax2 = ax1.twinx()
    sns.lineplot(
        x=errors["Date"], y=errors["Mean Training RMSE(£)"], marker="o", ax=ax1
    )
    sns.lineplot(x=errors["Date"], y=errors["Mean Training R2(%)"], marker="*", ax=ax2)
    st.pyplot(fig)


def app_view():

    title = "Analysing work orders data"
    st.set_page_config(page_title=title, layout="wide")

    st.header(title)

    X, y, rows = setup_inputs()
    if (X is not None) & (y is not None):
        dataset_raw = pd.concat([X, y], axis=1)
        dataset = pd.concat([X, np.log(y)], axis=1).rename(
            columns={y.columns[0]: f"LOG({y.columns[0]})"}
        )

        plot_counts(dataset_raw, rows)
        plot_relationships(dataset, f"LOG({y.columns[0]})")
        st.subheader("Training a model to predict cost")
        estimator, report, feature_imps = show_model(X, y)

        exp = st.beta_expander("View summary of training results")
        exp.table(report)

        st.bar_chart(feature_imps)
    else:
        estimator = get_existing_model()

    if estimator is not None:
        user_test_data(estimator)
        load_error_history()


if __name__ == "__main__":
    app_view()
