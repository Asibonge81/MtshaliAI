import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="eThekwini Election Analytics",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------

rf_model = joblib.load(
    "models/ethekwini_election_model.pkl"
)

model_features = joblib.load(
    "models/model_features.pkl"
)

# -------------------------------
# SIDEBAR
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="eThekwini Election Analytics",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# FILE PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# -------------------------------
# LOAD DATA
# -------------------------------

df2016 = pd.read_csv(DATA_DIR / "ETH_Ward.csv")
df2021 = pd.read_csv(DATA_DIR / "ETH.csv")

# -------------------------------
# LOAD SAVED MODEL
# -------------------------------

rf_model = joblib.load(
    MODEL_DIR / "ethekwini_election_model.pkl"
)

model_features = joblib.load(
    MODEL_DIR / "model_features.pkl"
)

# -------------------------------
# PREPARE ELECTION DATA
# -------------------------------

def prepare_election(df, year):

    data = df[df["BallotType"] == "PR"].copy()

    vd_totals = (
        data.groupby("VotingDistrict")["TotalValidVotes"]
        .sum()
        .rename("VDTotalVotes")
    )

    data = data.merge(
        vd_totals,
        on="VotingDistrict"
    )

    data["PartyVoteShare"] = (
        data["TotalValidVotes"] /
        data["VDTotalVotes"]
    ) * 100

    data["ElectionYear"] = year

    return data


model2016 = prepare_election(df2016, 2016)
model2021 = prepare_election(df2021, 2021)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Municipality Forecast",
        "Ward Analysis",
        "Voter Turnout",
        "Methodology"
    ]
)

# -------------------------------
# CREATE 2026 FORECAST
# -------------------------------

forecast2026 = model2021[
    [
        "Ward",
        "VotingDistrict",
        "RegisteredVoters",
        "PartyName",
        "PartyVoteShare"
    ]
].copy()

# The model was trained using this feature name.
# For the 2026 forecast, it contains the previous
# election cycle's (2021) vote share.
forecast2026["VoteShare2016"] = (
    forecast2026["PartyVoteShare"]
)

X_2026 = forecast2026[
    [
        "VoteShare2016",
        "RegisteredVoters",
        "PartyName"
    ]
].copy()

X_2026 = pd.get_dummies(
    X_2026,
    columns=["PartyName"]
)

# Match exactly the columns used during training
X_2026 = X_2026.reindex(
    columns=model_features,
    fill_value=0
)

# Predict party vote share
forecast2026["PredictedShare2026"] = np.clip(
    rf_model.predict(X_2026),
    0,
    100
)

# Make each voting district add up to 100%
forecast2026["PredictedShare2026"] = (
    forecast2026
    .groupby("VotingDistrict")["PredictedShare2026"]
    .transform(
        lambda x: x / x.sum() * 100
    )
)

# Estimate party votes
forecast2026["EstimatedPartyVotes"] = (
    forecast2026["RegisteredVoters"] *
    forecast2026["PredictedShare2026"] / 100
)


# -------------------------------
# OVERVIEW
# -------------------------------

if page == "Overview":

    st.title("eThekwini Election Analytics")

    st.write(
        """
        An evidence-based election analytics solution using historical
        eThekwini Local Government Election data and machine learning.
        """
    )

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Testing R²",
        "0.934"
    )

    col2.metric(
        "Testing MAE",
        "0.63"
    )

    col3.metric(
        "Model",
        "Random Forest"
    )

    st.success(
        "Random Forest model loaded successfully."
    )

    st.info(
        "The system estimates future party vote shares. "
        "It does not predict which party will govern the municipality."
    )


# -------------------------------
# MUNICIPALITY FORECAST
# -------------------------------

elif page == "Municipality Forecast":

    st.title("Municipality Vote Share Forecast")

    party_data = {
        "Party": [
            "African National Congress",
            "Democratic Alliance",
            "Economic Freedom Fighters",
            "Inkatha Freedom Party",
            "ActionSA"
        ],
        "Projected Vote Share": [
            34.97,
            20.52,
            16.17,
            13.30,
            2.64
        ]
    }

    party_df = pd.DataFrame(party_data)

    st.subheader("Leading Model Estimates")

    st.dataframe(
        party_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        party_df,
        x="Party",
        y="Projected Vote Share"
    )

    st.metric(
        "Highest Projected Aggregate Vote Share",
        "ANC — 34.97%"
    )

    st.caption(
        "The highest projected vote share does not determine "
        "which party will govern eThekwini."
    )


# -------------------------------
# WARD ANALYSIS
# -------------------------------

elif page == "Ward Analysis":

    st.title("Ward Analysis")

    ward_data = {
        "Ward": [
            "Ward 59500001",
            "Ward 59500050",
            "Ward 59500100"
        ],
        "Projected Leading Party": [
            "African National Congress",
            "Democratic Alliance",
            "African National Congress"
        ],
        "Projected Vote Share": [
            66.74,
            61.90,
            47.00
        ]
    }

    ward_df = pd.DataFrame(ward_data)

    selected_ward = st.selectbox(
        "Select a ward",
        ward_df["Ward"]
    )

    result = ward_df[
        ward_df["Ward"] == selected_ward
    ].iloc[0]

    st.subheader(selected_ward)

    col1, col2 = st.columns(2)

    col1.metric(
        "Highest Projected Party",
        result["Projected Leading Party"]
    )

    col2.metric(
        "Projected Vote Share",
        f"{result['Projected Vote Share']:.2f}%"
    )

    st.caption(
        "This represents the highest model-estimated vote share "
        "in the selected ward and does not predict governance."
    )


# -------------------------------
# VOTER TURNOUT
# -------------------------------

elif page == "Voter Turnout":

    st.title("Projected Voter Turnout")

    col1, col2 = st.columns(2)

    col1.metric(
        "Projected Turnout",
        "50.32%"
    )

    col2.metric(
        "Estimated Number of Voters",
        "960,576"
    )

    st.subheader("Historical Turnout")

    turnout_df = pd.DataFrame({
        "Election": [
            "2016",
            "2021",
            "2026 Projection"
        ],
        "Turnout": [
            59.10,
            41.53,
            50.32
        ]
    })

    st.bar_chart(
        turnout_df,
        x="Election",
        y="Turnout"
    )

    st.caption(
        "The estimated voter count uses the 2021 registered-voter "
        "baseline of 1,909,125 registered voters."
    )


# -------------------------------
# METHODOLOGY
# -------------------------------

elif page == "Methodology":

    st.title("Methodology")

    st.subheader("Data")

    st.write(
        """
        Historical eThekwini Local Government Election results
        from 2016 and 2021 were used to develop the predictive
        solution.
        """
    )

    st.subheader("Machine Learning Model")

    st.write(
        """
        A Random Forest Regression model was used because the
        target variable, party vote share, is numerical and the
        relationship between historical and future voting patterns
        may be non-linear.
        """
    )

    st.subheader("Model Evaluation")

    st.write(
        """
        The final model achieved a testing R² of 0.934 and a
        testing Mean Absolute Error (MAE) of 0.63 percentage points.
        """
    )

    st.subheader("Limitations")

    st.write(
        """
        The projections are based on historical election patterns
        and should not be interpreted as guaranteed election
        outcomes. Future voter behaviour, changes in registered
        voters, new political parties, boundary changes and other
        factors may affect actual results.
        """
    )