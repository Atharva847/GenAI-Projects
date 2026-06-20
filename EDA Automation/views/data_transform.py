import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
    SequentialFeatureSelector,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Lasso, LogisticRegression


def _init_transform_state(df):
    df_key = f"transformed_df_{len(df)}_{list(df.columns)}"
    if "transformed_df" not in st.session_state or st.session_state.get("df_key") != df_key:
        st.session_state.transformed_df = df.copy()
        st.session_state.df_key = df_key
        st.session_state.transform_history = []
        st.session_state.dup_handled = False


def _record(action):
    if "transform_history" not in st.session_state:
        st.session_state.transform_history = []
    st.session_state.transform_history.append(action)


def _is_classification_target(y):
    return y.dtype == "object" or y.dtype.name == "category" or y.nunique() <= 20


def render(df):
    _init_transform_state(df)

    _handle_drop_columns()
    st.markdown("---")
    _handle_duplicates()
    st.markdown("---")
    _handle_missing_values()
    st.markdown("---")
    _handle_encoding()
    st.markdown("---")
    _handle_scaling()
    st.markdown("---")
    _handle_feature_selection()


def _handle_drop_columns():
    st.subheader("🗑️ Drop Columns")
    current_df = st.session_state.transformed_df
    all_cols = current_df.columns.tolist()
    st.caption("Select one or more columns to permanently remove from the dataset.")
    cols_to_drop = st.multiselect("Select columns to drop", options=all_cols, key="drop_cols_select")

    if cols_to_drop:
        st.warning(f"⚠️ You are about to drop {len(cols_to_drop)} column(s): **{', '.join(cols_to_drop)}**")
        if st.button("Apply Drop Columns", key="apply_drop_cols"):
            with st.spinner("Dropping selected columns..."):
                time.sleep(1.5)
                working_df = st.session_state.transformed_df.copy()
                working_df = working_df.drop(columns=cols_to_drop)
                st.session_state.transformed_df = working_df
                _record({"type": "drop_columns", "columns": cols_to_drop})
            st.success(f"✅ Successfully dropped: **{', '.join(cols_to_drop)}**")
            st.markdown(f"**Remaining Columns:** {len(st.session_state.transformed_df.columns)}")
            st.dataframe(st.session_state.transformed_df.head(10), use_container_width=True)
    else:
        st.info("ℹ️ No columns selected. Select columns above to drop them.")


def _handle_duplicates():
    st.subheader("🔄 Handle Duplicates")
    df_copy = st.session_state.transformed_df
    dup = df_copy.duplicated().sum()

    if dup > 0:
        if not st.session_state.dup_handled:
            if st.button("Handle Duplicates"):
                st.session_state.dup_handled = True

        if st.session_state.dup_handled:
            st.metric("Total Duplicate Rows", dup)
            choice = st.radio("Duplicate rows found. Do you want to remove them?", ["Yes", "No"], index=None)

            if choice == "Yes":
                with st.spinner("In progress..."):
                    time.sleep(1.5)
                    df_copy = df_copy.drop_duplicates()
                    st.session_state.transformed_df = df_copy
                    _record({"type": "drop_duplicates"})
                    remaining_dup = df_copy.duplicated().sum()
                st.success("✅ Successfully Dropped Duplicate Rows")
                st.markdown(f"Total Number of Duplicate Values : :blue-background[{remaining_dup}]")
                st.dataframe({'Columns': df_copy.columns, 'Duplicates': remaining_dup})
            elif choice == "No":
                st.info("❎ Duplicate rows were not removed.")
                st.markdown(f"Total Number of Duplicate Values : :blue-background[{dup}]")
                st.dataframe(df_copy)
    else:
        st.success("✅ No Duplicate Rows Detected.")


def _handle_missing_values():
    st.subheader("🩹 Handle Missing Values")
    current_df = st.session_state.transformed_df
    total_missing = current_df.isnull().sum().sum()

    if total_missing == 0:
        st.success("✅ No Missing Values Detected.")
        return

    missing_df = current_df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    missing_df["Missing %"] = ((missing_df["Missing Count"] / len(current_df)) * 100).round(2)
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    st.markdown(f"**Total Missing Values:** :red[{total_missing}]")
    st.dataframe(missing_df, use_container_width=True)
    st.markdown("---")

    num_missing_cols = current_df[missing_df["Column"].tolist()].select_dtypes(
        include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_missing_cols = current_df[missing_df["Column"].tolist()].select_dtypes(
        include=['object', 'category']).columns.tolist()

    fill_strategy = {}

    if num_missing_cols:
        st.markdown("#### 🔢 Numeric Columns — Choose Fill Strategy")
        st.caption("Select Mean or Median for each numeric column that has missing values.")
        for col in num_missing_cols:
            col_missing = int(current_df[col].isnull().sum())
            c1, c2 = st.columns([2, 1])
            with c1:
                strategy = st.radio(
                    f"`{col}` — {col_missing} missing ({round(col_missing / len(current_df) * 100, 2)}%)",
                    ["Mean", "Median"], key=f"fill_num_{col}", horizontal=True
                )
            with c2:
                st.markdown(f"Mean: **{round(current_df[col].mean(), 4)}** | Median: **{round(current_df[col].median(), 4)}**")
            fill_strategy[col] = strategy

    if cat_missing_cols:
        st.markdown("#### 🔤 Categorical Columns — Fill with Mode")
        st.caption("Categorical missing values will be automatically filled with the most frequent value (Mode).")
        mode_preview = []
        for col in cat_missing_cols:
            col_missing = int(current_df[col].isnull().sum())
            mode_val = current_df[col].mode()[0] if not current_df[col].mode().empty else "N/A"
            mode_preview.append({
                "Column": col, "Missing Count": col_missing,
                "Missing %": round(col_missing / len(current_df) * 100, 2),
                "Mode (Fill Value)": mode_val
            })
            fill_strategy[col] = "Mode"
        st.dataframe(pd.DataFrame(mode_preview), use_container_width=True)

    st.markdown("---")
    if st.button("Apply Missing Value Treatment", key="apply_missing"):
        with st.spinner("Filling missing values..."):
            time.sleep(1.5)
            working_df = st.session_state.transformed_df.copy()
            for col, strategy in fill_strategy.items():
                if strategy == "Mean":
                    working_df[col] = working_df[col].fillna(working_df[col].mean())
                elif strategy == "Median":
                    working_df[col] = working_df[col].fillna(working_df[col].median())
                elif strategy == "Mode":
                    mode_val = working_df[col].mode()[0] if not working_df[col].mode().empty else np.nan
                    working_df[col] = working_df[col].fillna(mode_val)
            st.session_state.transformed_df = working_df
            _record({"type": "missing_values", "strategies": fill_strategy.copy()})

        remaining_missing = st.session_state.transformed_df.isnull().sum().sum()
        st.success("✅ Missing Values Filled Successfully!")
        st.markdown(f"**Remaining Missing Values:** :blue-background[{remaining_missing}]")
        st.dataframe(st.session_state.transformed_df.head(10), use_container_width=True)


def _handle_encoding():
    st.subheader("🏷️ Categorical Encoding")
    cat_cols = st.session_state.transformed_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        st.warning("⚠️ No categorical columns found to encode.")
        return

    st.caption("Choose an encoding method per column — mix Label Encoding and One-Hot Encoding as needed.")
    encoding_plan = {}
    for col in cat_cols:
        encoding_plan[col] = st.selectbox(
            f"`{col}`",
            ["Skip", "Label Encoding", "One-Hot Encoding"],
            key=f"enc_method_{col}",
        )

    label_cols = [c for c, m in encoding_plan.items() if m == "Label Encoding"]
    ohe_cols = [c for c, m in encoding_plan.items() if m == "One-Hot Encoding"]

    if not label_cols and not ohe_cols:
        st.info("ℹ️ Select at least one column to encode (choose Label or One-Hot above).")
        return

    le_add_new = False
    if label_cols:
        le_mode = st.radio(
            "Label Encoding — replace original columns or add new encoded columns?",
            ["Replace original columns", "Add new columns (_Encoded)"],
            key="le_mode",
            horizontal=True,
        )
        le_add_new = le_mode == "Add new columns (_Encoded)"

    ohe_drop_first = False
    ohe_drop_original = False
    if ohe_cols:
        st.markdown("#### One-Hot Encoding options")
        for col in ohe_cols:
            st.write(f"Unique values in `{col}`:", st.session_state.transformed_df[col].unique().tolist())
        ohe_drop_first = st.checkbox(
            "Drop first category column per feature (sklearn `drop='first'`)",
            key="ohe_drop_first",
        )
        drop_orig_choice = st.radio(
            "Drop original columns after encoding?",
            ["Yes", "No"],
            key="ohe_drop_orig",
            horizontal=True,
        )
        ohe_drop_original = drop_orig_choice == "Yes"

    if st.button("Apply Encoding", key="apply_encoding"):
        with st.spinner("Applying encoding..."):
            time.sleep(1.5)
            working_df = st.session_state.transformed_df.copy()

            for col in label_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(working_df[col].astype(str))
                if le_add_new:
                    working_df[f"{col}_Encoded"] = encoded
                else:
                    working_df[col] = encoded

            if ohe_cols:
                drop_param = "first" if ohe_drop_first else None
                ohe = OneHotEncoder(sparse_output=False, drop=drop_param, handle_unknown="ignore")
                transformed_data = ohe.fit_transform(working_df[ohe_cols].astype(str))
                encoded_col_names = ohe.get_feature_names_out(ohe_cols)
                encoded_df = pd.DataFrame(transformed_data, columns=encoded_col_names, index=working_df.index)
                if ohe_drop_original:
                    working_df = working_df.drop(columns=ohe_cols)
                working_df = pd.concat([working_df, encoded_df], axis=1)

            st.session_state.transformed_df = working_df
            _record({
                "type": "custom_encoding",
                "label_columns": label_cols,
                "label_add_new": le_add_new,
                "ohe_columns": ohe_cols,
                "ohe_drop_first": ohe_drop_first,
                "ohe_drop_original": ohe_drop_original,
            })

        st.success("✅ Encoding applied successfully!")
        st.subheader("🔍 Transformed Data Preview")
        st.dataframe(st.session_state.transformed_df.head(), use_container_width=True)


def _handle_scaling():
    st.subheader("📏 Feature Scaling")
    current_df = st.session_state.transformed_df
    num_cols = current_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

    if not num_cols:
        st.warning("⚠️ No Numerical columns found.")
        return

    multi_sel = st.multiselect("Select Features", num_cols, key="scale_features")
    if not multi_sel:
        st.info("ℹ️ Select one or more numeric features to scale.")
        return

    scale_type = st.selectbox(
        "Select Feature Scaling type",
        ["Select an option", "Standardization", "MinMaxScaling", "Robust Scaling"],
        key="scale_type",
    )

    if scale_type == "Select an option":
        return

    if st.button("Apply Feature Scaling", key="apply_scaling"):
        with st.spinner(f"Applying {scale_type}..."):
            time.sleep(1.5)
            scaled_df = current_df.copy()
            if scale_type == "Standardization":
                scaler = StandardScaler()
                method = "Standardization"
                success_msg = "✅ Standardization (Z-score) applied successfully!"
                desc = "**Mean ≈ 0, Std ≈ 1 for selected features**"
            elif scale_type == "MinMaxScaling":
                scaler = MinMaxScaler()
                method = "MinMaxScaling"
                success_msg = "✅ MinMax Scaling applied successfully! Values normalized to [0, 1]"
                desc = ""
            else:
                scaler = RobustScaler()
                method = "Robust Scaling"
                success_msg = "✅ Robust Scaling applied successfully! Scaled using median and IQR (outlier-resistant)"
                desc = ""

            scaled_values = scaler.fit_transform(current_df[multi_sel])
            scaled_df[multi_sel] = scaled_values
            st.session_state.transformed_df = scaled_df
            _record({"type": "scaling", "method": method, "columns": multi_sel})

        st.success(success_msg)
        if desc:
            st.markdown(desc)
        st.dataframe(scaled_df[multi_sel].describe().round(4), use_container_width=True)
        st.dataframe(scaled_df.head(), use_container_width=True)


def _prepare_xy(current_df, target_col, feature_cols):
    X = current_df[feature_cols].copy()
    y = current_df[target_col].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if _is_classification_target(y):
        if y.dtype == "object" or y.dtype.name == "category":
            y = LabelEncoder().fit_transform(y.astype(str))
        is_clf = True
    else:
        y = pd.to_numeric(y, errors="coerce")
        is_clf = False

    # Chi2 needs non-negative values
    X_nonneg = X.copy()
    for col in X_nonneg.columns:
        min_val = X_nonneg[col].min()
        if min_val < 0:
            X_nonneg[col] = X_nonneg[col] - min_val

    return X, X_nonneg, y, is_clf


def _handle_feature_selection():
    st.subheader("🎯 Feature Selection")
    current_df = st.session_state.transformed_df
    all_cols = current_df.columns.tolist()

    method_category = st.selectbox(
        "Select method category",
        ["Select", "Filter Methods", "Wrapper Methods", "Embedded Methods"],
        key="fs_category",
    )
    if method_category == "Select":
        st.info("ℹ️ Choose a feature selection category to get started.")
        return

    target_col = st.selectbox("Select target column", ["Select"] + all_cols, key="fs_target")
    if target_col == "Select":
        st.warning("⚠️ Please select a target column.")
        return

    feature_cols = [c for c in all_cols if c != target_col]
    if not feature_cols:
        st.warning("⚠️ No feature columns available.")
        return

    X, X_nonneg, y, is_clf = _prepare_xy(current_df, target_col, feature_cols)
    selected_features = None
    scores_df = None

    if method_category == "Filter Methods":
        filter_method = st.selectbox(
            "Filter method",
            ["Variance Threshold", "Chi-Square", "Mutual Information", "Correlation with Target"],
            key="filter_method",
        )

        if filter_method == "Variance Threshold":
            threshold = st.slider("Variance threshold", 0.0, 1.0, 0.0, 0.01, key="var_thresh")
            if st.button("Apply Filter Selection", key="apply_filter_var"):
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(X)
                mask = selector.get_support()
                selected_features = [feature_cols[i] for i, keep in enumerate(mask) if keep]
                _apply_feature_selection(selected_features, target_col, "Filter", filter_method, {"threshold": threshold})

        elif filter_method == "Chi-Square":
            k = st.slider("Number of top features (k)", 1, len(feature_cols), min(5, len(feature_cols)), key="chi2_k")
            if st.button("Apply Filter Selection", key="apply_filter_chi2"):
                selector = SelectKBest(score_func=chi2, k=k)
                selector.fit(X_nonneg, y)
                mask = selector.get_support()
                selected_features = [feature_cols[i] for i, keep in enumerate(mask) if keep]
                scores_df = pd.DataFrame({"Feature": feature_cols, "Chi2 Score": selector.scores_}).sort_values(
                    "Chi2 Score", ascending=False
                )
                _apply_feature_selection(selected_features, target_col, "Filter", filter_method, {"k": k})

        elif filter_method == "Mutual Information":
            k = st.slider("Number of top features (k)", 1, len(feature_cols), min(5, len(feature_cols)), key="mi_k")
            score_func = mutual_info_classif if is_clf else mutual_info_regression
            if st.button("Apply Filter Selection", key="apply_filter_mi"):
                selector = SelectKBest(score_func=score_func, k=k)
                selector.fit(X, y)
                mask = selector.get_support()
                selected_features = [feature_cols[i] for i, keep in enumerate(mask) if keep]
                scores_df = pd.DataFrame({"Feature": feature_cols, "MI Score": selector.scores_}).sort_values(
                    "MI Score", ascending=False
                )
                _apply_feature_selection(selected_features, target_col, "Filter", filter_method, {"k": k})

        elif filter_method == "Correlation with Target":
            threshold = st.slider("Min |correlation| with target", 0.0, 1.0, 0.1, 0.05, key="corr_thresh")
            if st.button("Apply Filter Selection", key="apply_filter_corr"):
                corrs = X.corrwith(pd.Series(y, index=X.index)).abs()
                selected_features = corrs[corrs >= threshold].index.tolist()
                scores_df = corrs.reset_index()
                scores_df.columns = ["Feature", "|Correlation|"]
                scores_df = scores_df.sort_values("|Correlation|", ascending=False)
                _apply_feature_selection(selected_features, target_col, "Filter", filter_method, {"threshold": threshold})

    elif method_category == "Wrapper Methods":
        wrapper_method = st.selectbox(
            "Wrapper method",
            ["Recursive Feature Elimination (RFE)", "Forward Selection", "Backward Elimination"],
            key="wrapper_method",
        )
        n_features = st.slider(
            "Number of features to select",
            1, len(feature_cols), min(5, len(feature_cols)), key="wrapper_n",
        )

        if st.button("Apply Wrapper Selection", key="apply_wrapper"):
            estimator = (
                RandomForestClassifier(n_estimators=50, random_state=42)
                if is_clf
                else RandomForestRegressor(n_estimators=50, random_state=42)
            )
            if wrapper_method == "Recursive Feature Elimination (RFE)":
                selector = RFE(estimator=estimator, n_features_to_select=n_features)
                selector.fit(X, y)
                mask = selector.support_
                selected_features = [feature_cols[i] for i, keep in enumerate(mask) if keep]
                _apply_feature_selection(
                    selected_features, target_col, "Wrapper", wrapper_method, {"n_features": n_features}
                )
            else:
                direction = "forward" if wrapper_method == "Forward Selection" else "backward"
                selector = SequentialFeatureSelector(
                    estimator=estimator,
                    n_features_to_select=n_features,
                    direction=direction,
                    cv=3,
                )
                selector.fit(X, y)
                mask = selector.get_support()
                selected_features = [feature_cols[i] for i, keep in enumerate(mask) if keep]
                _apply_feature_selection(
                    selected_features, target_col, "Wrapper", wrapper_method,
                    {"n_features": n_features, "direction": direction},
                )

    elif method_category == "Embedded Methods":
        embedded_method = st.selectbox(
            "Embedded method",
            ["Lasso (L1)", "Random Forest Importance", "Decision Tree Importance"],
            key="embedded_method",
        )
        threshold = st.slider("Importance / coefficient threshold", 0.0, 1.0, 0.01, 0.01, key="embed_thresh")

        if st.button("Apply Embedded Selection", key="apply_embedded"):
            if embedded_method == "Lasso (L1)":
                if is_clf:
                    estimator = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
                else:
                    estimator = Lasso(alpha=0.01, random_state=42)
            elif embedded_method == "Random Forest Importance":
                estimator = (
                    RandomForestClassifier(n_estimators=50, random_state=42)
                    if is_clf
                    else RandomForestRegressor(n_estimators=50, random_state=42)
                )
            else:
                estimator = (
                    DecisionTreeClassifier(random_state=42)
                    if is_clf
                    else DecisionTreeRegressor(random_state=42)
                )

            selector = SelectFromModel(estimator, threshold=threshold)
            selector.fit(X, y)
            mask = selector.get_support()
            selected_features = [feature_cols[i] for i, keep in enumerate(mask) if keep]

            if hasattr(estimator, "feature_importances_"):
                scores_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": estimator.feature_importances_,
                }).sort_values("Importance", ascending=False)
            elif hasattr(estimator, "coef_"):
                coef = estimator.coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                scores_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": np.abs(coef),
                }).sort_values("Coefficient", ascending=False)

            _apply_feature_selection(
                selected_features, target_col, "Embedded", embedded_method, {"threshold": threshold}
            )

    if scores_df is not None:
        st.markdown("**Feature scores**")
        st.dataframe(scores_df, use_container_width=True)


def _apply_feature_selection(selected_features, target_col, category, method, params):
    if not selected_features:
        st.warning("⚠️ No features met the selection criteria. Try adjusting parameters.")
        return

    keep_cols = selected_features + [target_col]
    with st.spinner("Applying feature selection..."):
        time.sleep(1.5)
        working_df = st.session_state.transformed_df[keep_cols].copy()
        st.session_state.transformed_df = working_df
        _record({
            "type": "feature_selection",
            "category": category,
            "method": method,
            "params": params,
            "selected_features": selected_features,
            "target": target_col,
        })

    st.success(f"✅ Selected **{len(selected_features)}** feature(s): {', '.join(selected_features)}")
    st.dataframe(st.session_state.transformed_df.head(10), use_container_width=True)
