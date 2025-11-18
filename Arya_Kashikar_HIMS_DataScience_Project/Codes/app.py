import streamlit as st
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO

# ML / stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional: set seaborn style
sns.set(style="whitegrid")

# Set page config as first Streamlit command
st.set_page_config(page_title="HMIS - EDA & Modeling", layout="wide")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def compute_health_index(df, top_indicators, state_col="State"):
    missing = [c for c in top_indicators if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for health index: {missing}")
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[top_indicators] = scaler.fit_transform(df_scaled[top_indicators])
    state_health_index = df_scaled.groupby(state_col)[top_indicators].mean()
    state_health_index['Health_Index'] = state_health_index.mean(axis=1)
    state_health_index_sorted = state_health_index[['Health_Index']].sort_values(by='Health_Index', ascending=False)
    return state_health_index_sorted

def to_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def main():
    st.title(" HMIS Analytics Dashboard")

    fixed_path = r"C:\Users\91971\Desktop\22070521036_Arya_Kashikar_HIMS_DataScience_Project\dataset\HMIS_Cleaned.csv"

    st.sidebar.header("Data & View Controls")

    data_source = st.sidebar.radio("Data Source", ("Use Fixed HMIS_Cleaned.csv", "Upload CSV"))
    if data_source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload HMIS_Cleaned.csv", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Dataset loaded from upload.")
        else:
            st.sidebar.info("Awaiting file upload.")
            df = None
    else:
        try:
            df = load_csv(fixed_path)
            st.sidebar.success(f"Loaded dataset from fixed path: {fixed_path}")
        except Exception as e:
            st.sidebar.error(f"Failed to load fixed dataset: {e}")
            df = None

    if df is not None:
        if "State" in df.columns:
            states = sorted(df["State"].dropna().unique())
            selected_states = st.sidebar.multiselect("States", states, default=states[:5])
            df = df[df["State"].isin(selected_states)]
        else:
            st.sidebar.warning("Column 'State' not found; skipping state filter.")

        if "YearMonth" in df.columns or ("Year" in df.columns and "Month" in df.columns):
            if "YearMonth" in df.columns:
                df["YearMonth"] = pd.to_datetime(df["YearMonth"])
            else:
                df["YearMonth"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
            min_date, max_date = df["YearMonth"].min(), df["YearMonth"].max()
            date_range = st.sidebar.date_input("Date range", [min_date, max_date])
            if date_range and len(date_range) == 2:
                start, end = date_range
                df = df[(df["YearMonth"] >= pd.to_datetime(start)) & (df["YearMonth"] <= pd.to_datetime(end))]

        st.sidebar.header("Clustering (Final Reg) Controls")
        clustering_method = st.sidebar.selectbox("Clustering method", ["KMeans", "Agglomerative"])
        n_clusters = st.sidebar.slider("Number of clusters", 2, 6, 3)

        reg_target = st.sidebar.selectbox("Regression target", [
            "In-Patient_Head_Count_At_Midnight_(UOM:Number),_Scaling_Factor:1"
        ])

        view_choice = st.sidebar.radio("View", ["EDA Health Index", "Modeling & Clustering"])

    else:
        st.error("Dataset not loaded. Please upload a CSV or fix the path.")
        return

    left, right = st.columns([3, 2])

    if view_choice == "EDA Health Index":
        with left:
            st.header("Health Index by State (Top Indicators)")
            top_indicators = [
                'Number_Of_Lab_Tests_Done__(UOM:Number),_Scaling_Factor:1',
                'Radiology_-_X-Ray__(UOM:Number),_Scaling_Factor:1',
                'Number_Of_Allopathic_Outpatient_Attendance__(UOM:Number),_Scaling_Factor:1',
                'In-Patient_Head_Count_At_Midnight_(UOM:Number),_Scaling_Factor:1',
                'Number_Of_Ayush_Outpatient_Attendance__(UOM:Number),_Scaling_Factor:1'
            ]
            try:
                health_index_df = compute_health_index(df, top_indicators, state_col="State")
                st.write(health_index_df.reset_index().rename(columns={"State": "State", "Health_Index": "Health Index"}))
                fig = px.bar(health_index_df.reset_index(),
                             x="Health_Index",
                             y=health_index_df.index,
                             orientation="h",
                             labels={"Health_Index": "Health Index", "index": "State"},
                             color_discrete_sequence=["#2ca02c"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error computing health index: {e}")

        with right:
            st.subheader("Data Preview")
            st.write(df.head())
            st.write(f"Dataset shape: {df.shape}")

            if "State" in df.columns:
                dist = df.groupby("State").size().sort_values(ascending=False).head(10)
                st.bar_chart(dist)

    else:
        with left:
            st.header("Clustering & Regression - Maharashtra Focus (example)")
            st.write("This view covers clustering and regression analyses from final_reg_classi_cluster.")
            st.markdown("Representative plots and metrics are shown below with interactive controls.")

            if "State" in df.columns:
                col_name = "In-Patient_Head_Count_At_Midnight_(UOM:Number),_Scaling_Factor:1"
                if col_name in df.columns:
                    cap = df.groupby("State")[col_name].mean().sort_values(ascending=False)
                    # Clean column names for plotting
                    cap_df = cap.reset_index()
                    cap_df.columns = ["State", "Mean_InPatient_HeadCount"]
                    st.write(cap_df.head(10))
                    st.line_chart(cap_df.set_index("State"))

        with right:
            st.subheader("Modeling Output (simplified)")
            st.write("This section demonstrates a streamlined modeling flow using the extracted blocks.")
            if "State" in df.columns:
                states = ["Maharashtra"]
                df_maha = df[df["State"].isin(states)]
                if not df_maha.empty:
                    features = [
                        'Number_Of_Lab_Tests_Done__(UOM:Number),_Scaling_Factor:1',
                        'Number_Of_Allopathic_Outpatient_Attendance__(UOM:Number),_Scaling_Factor:1',
                        'Number_Of_Condom_Pieces_Distributed_(UOM:Number),_Scaling_Factor:1',
                        'Number_Of_Haemoglobin_(Hb)_Tests_Conducted__(UOM:Number),_Scaling_Factor:1',
                        'Patients_Registered_At_Emergency_Department_(UOM:Number),_Scaling_Factor:1',
                        'In-Patient_Head_Count_At_Midnight_(UOM:Number),_Scaling_Factor:1'
                    ]

                    if all(c in df.columns for c in features) and reg_target in df.columns:
                        X = df[features].dropna()
                        y = df.loc[X.index, reg_target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        rf = RandomForestRegressor(random_state=42)
                        rf.fit(X_train, y_train)
                        preds = rf.predict(X_test)
                        mae = mean_absolute_error(y_test, preds)
                        rmse = mean_squared_error(y_test, preds, squared=False)
                        r2 = r2_score(y_test, preds)
                        st.write("Random Forest Regression (Maharashtra):")
                        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.3f}")
                        st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": preds}).reset_index(drop=True))
                    else:
                        st.info("Required features or target not found for regression in Maharashtra subset.")
                else:
                    st.info("No Maharashtra data found in the current filtered dataset.")

    if df is not None:
        st.sidebar.header("Export")
        export_df = df.copy()
        buf = to_csv_bytes(export_df)
        st.sidebar.download_button(label="Download current dataset (CSV)",
                                   data=buf.getvalue(),
                                   file_name="HMIS_Cleaned_processed.csv",
                                   mime="text/csv")

if __name__ == "__main__":
    main()
