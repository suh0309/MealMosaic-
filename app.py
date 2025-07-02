import os
import io
import math 
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="MealMosaic Analytics", layout="wide")

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

DATA_PATH = os.path.join(os.path.dirname(__file__), "mealmosaic_survey_dataset.csv")
df = load_data(DATA_PATH)

st.title("📊 MealMosaic Consumer Insights Dashboard")

# ----------- SIDEBAR -------------
with st.sidebar:
    st.header("Global Filters")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    with st.expander("Subset columns"):
        selected_cols = st.multiselect("Columns to include", df.columns, default=df.columns)
        df = df[selected_cols]

    st.write("After adjusting filters, choose a tab to begin ➡️")

# ----------- MAIN TABS -----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Visualisation", "Classification", "Clustering", "Association Rules", "Regression"]
)

# ---------- TAB 1: VISUALISATION ----------
with tab1:
    st.subheader("Descriptive Insights")
    st.write("Below are 10 quick insights drawn from the current dataset view.")
    if df.empty:
        st.warning("No data to display – adjust your filters in the sidebar.")
    else:
        # 1. Age distribution
        st.markdown("**1. Age Distribution**")
        fig, ax = plt.subplots()
        ax.hist(df["Age"], bins=20)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # 2. Typical Spend distribution
        st.markdown("**2. Typical Spend (₹) Distribution**")
        fig, ax = plt.subplots()
        ax.hist(df["TypicalSpend"], bins=30)
        ax.set_xlabel("Spend (₹)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # 3. Income vs Typical Spend scatter
        if "IncomeBracket" in df.columns:
            st.markdown("**3. Typical Spend vs. Income Bracket**")
            income_map = {'<25k':1, '25-50k':2, '50-75k':3, '75-100k':4, '>100k':5}
            df_tmp = df.copy()
            df_tmp["IncomeCode"] = df_tmp["IncomeBracket"].map(income_map)
            fig, ax = plt.subplots()
            ax.scatter(df_tmp["IncomeCode"], df_tmp["TypicalSpend"])
            ax.set_xlabel("Income Bracket Code (low→high)")
            ax.set_ylabel("Typical Spend (₹)")
            st.pyplot(fig)

        # 4. Order frequency vs adoption
        st.markdown("**4. Order Frequency vs Adoption Likelihood**")
        if "OrderFreq" in df.columns:
            freq_order = ['<1/mo', '1-3/mo', '1-2/wk', '3-5/wk', '>5/wk']
            freq_map = {v: i for i, v in enumerate(freq_order)}
            df_tmp = df.copy()
            df_tmp["FreqCode"] = df_tmp["OrderFreq"].map(freq_map)
            fig, ax = plt.subplots()
            ax.scatter(df_tmp["FreqCode"], df_tmp["AdoptionLikelihood_30d"])
            ax.set_xlabel("Order Frequency Code (low→high)")
            ax.set_ylabel("Adoption Likelihood (1-5)")
            st.pyplot(fig)

        # 5. Heatmap of numeric correlations
        st.markdown("**5. Correlation Heatmap (numeric features)**")
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(corr, cmap='viridis')
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(im)
        st.pyplot(fig)

        # Summary stats table
        st.markdown("**6. Summary Statistics**")
        st.dataframe(df.describe())

        # 7-10 Additional quick stats
        st.markdown("**7–10. Additional quick facts**")
        col1, col2 = st.columns(2)
        with col1:
            avg_spend = df["TypicalSpend"].mean()
            st.metric("Average Spend (₹)", f"{avg_spend:,.0f}")
            high_spend = df["TypicalSpend"].quantile(0.95)
            st.metric("95th‑pct Spend", f"{high_spend:,.0f}")
        with col2:
            avg_age = df["Age"].mean()
            st.metric("Average Age", f"{avg_age:,.1f} yrs")
            adopters = (df["AdoptionLikelihood_30d"]>=4).mean()*100
            st.metric("Likely Adopters (%)", f"{adopters:.1f}%")

# ---------- UTILITIES ----------
def preprocess(df_in, target_col: str):
    df = df_in.copy()
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )
    return X, y, pre

# ---------- TAB 2: CLASSIFICATION ----------
with tab2:
    st.subheader("Classification Models – Predict Adoption Likelihood ≥ 4 (Yes/No)")
    # Create binary target
    df_cls = df.copy()
    df_cls["Adopt"] = (df_cls["AdoptionLikelihood_30d"] >= 4).astype(int)
    X, y, pre = preprocess(df_cls.drop(columns=["AdoptionLikelihood_30d"]), "Adopt")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }

    metrics_table = []
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics_table.append([name, f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fpr_dict[name] = fpr
        tpr_dict[name] = tpr
        auc_dict[name] = auc(fpr, tpr)

    st.markdown("**Performance Comparison**")
    mt_df = pd.DataFrame(metrics_table, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
    st.dataframe(mt_df)

    st.markdown("**Confusion Matrix**")
    selected_model = st.selectbox("Select model to view confusion matrix", list(models.keys()))
    sel_pipe = Pipeline(steps=[("pre", pre), ("model", models[selected_model])])
    sel_pipe.fit(X_train, y_train)
    cm = confusion_matrix(y_test, sel_pipe.predict(X_test))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["0","1"])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["0","1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    st.pyplot(fig)

    st.markdown("**ROC Curves**")
    fig, ax = plt.subplots()
    for name in models:
        pipe = Pipeline(steps=[("pre", pre), ("model", models[name])])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Predict on New Data**")
    up_file = st.file_uploader("Upload new CSV (without target columns)", type=["csv"])
    if up_file:
        new_df = pd.read_csv(up_file)
        pipe_final = Pipeline(steps=[("pre", pre), ("model", models[selected_model])])
        pipe_final.fit(X, y)
        preds = pipe_final.predict(new_df)
        new_df["Predicted_Adopt"] = preds
        st.dataframe(new_df.head())
        to_download = new_df.to_csv(index=False).encode()
        st.download_button("Download predictions", to_download, file_name="predictions.csv")

# ---------- TAB 3: CLUSTERING ----------
with tab3:
    st.subheader("Customer Segmentation – K‑Means")
    features_for_cluster = st.multiselect(
        "Select numeric features for clustering",
        df.select_dtypes(include=np.number).columns.tolist(),
        default=["Age", "TypicalSpend", "NutritionImportance", "BasketCombineInterest"]
    )
    if len(features_for_cluster) >= 2:
        data_scaled = StandardScaler().fit_transform(df[features_for_cluster])
        max_k = 10
        fig, ax = plt.subplots()
        distortions = []
        for k in range(2, max_k+1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(data_scaled)
            distortions.append(km.inertia_)
        ax.plot(range(2, max_k+1), distortions, marker="o")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Inertia (SSE)")
        st.markdown("**Elbow Chart**")
        st.pyplot(fig)

        cluster_k = st.slider("Select number of clusters", 2, 10, 4)
        km_final = KMeans(n_clusters=cluster_k, n_init=20, random_state=42)
        labels = km_final.fit_predict(data_scaled)
        df_clustered = df.copy()
        df_clustered["Cluster"] = labels

        st.markdown("**Cluster Personas (centroid values)**")
        centroids = pd.DataFrame(km_final.cluster_centers_, columns=features_for_cluster)
        st.dataframe(centroids.style.format("{:.2f}"))

        csv_with_labels = df_clustered.to_csv(index=False).encode()
        st.download_button("Download data with cluster labels", csv_with_labels,
                           file_name="clustered_data.csv")
    else:
        st.info("Select at least two numeric features for clustering.")

# ---------- TAB 4: ASSOCIATION RULES ----------
with tab4:
    st.subheader("Association Rule Mining – Apriori")
    multi_cols = [c for c in df.columns if df[c].dtype == object and df[c].str.contains(",").any()]
    if multi_cols:
        col_choice = st.selectbox("Select multi-select column for market-basket‑style analysis", multi_cols)
        df_basket = df[col_choice].str.get_dummies(sep=',')
        min_support = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
        frequent = apriori(df_basket, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values(by="confidence", ascending=False).head(10)
        st.markdown(f"**Top‑10 rules for {col_choice}**")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    else:
        st.info("No multi-select columns detected for association rule mining.")

# ---------- TAB 5: REGRESSION ----------
with tab5:
    st.subheader("Spend Prediction – Regression Models")
    target = "TypicalSpend"
    df_reg = df.copy()
    Xr, yr, pre_r = preprocess(df_reg, target)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.5),
        "Decision Tree": DecisionTreeRegressor(max_depth=6)
    }
    reg_metrics = []
    from sklearn.metrics import r2_score, mean_squared_error
    for name, mdl in reg_models.items():
        pipe = Pipeline(steps=[("pre", pre_r), ("model", mdl)])
        pipe.fit(Xr_train, yr_train)
        preds = pipe.predict(Xr_test)
        r2 = r2_score(yr_test, preds)
        rmse = math.sqrt(mean_squared_error(yr_test, preds))
        reg_metrics.append([name, f"{r2:.3f}", f"{rmse:.1f}"])

    st.markdown("**Performance Summary**")
    reg_df = pd.DataFrame(reg_metrics, columns=["Model", "R²", "RMSE"])
    st.dataframe(reg_df)

    st.markdown("**Insight:** Decision‑tree regressors often capture non‑linear spend patterns among niche segments (e.g., older high earners). Compare RMSE to judge which model best generalises.")
