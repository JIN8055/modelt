import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

MODEL_PATH = "models/best_model.pkl"
DATA_URL = "https://raw.githubusercontent.com/JIN8055/modelt/main/Data/titanic.csv"

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

@st.cache_data
def load_data(url=DATA_URL):
    df = pd.read_csv(url)
    if "Title" not in df.columns and "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r',\s*([^\.]*)\.', expand=False)
        df["Title"] = df["Title"].replace(
            ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
            'Rare'
        )
        df["Title"] = df["Title"].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    return df

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.error(f"Could not load model from {path}: {e}")
        return None

def make_title_options(df):
    if 'Title' in df.columns:
        opts = list(df['Title'].dropna().unique())
        preferred = ['Mr', 'Mrs', 'Miss', 'Master']
        others = [o for o in opts if o not in preferred]
        ordered = [o for o in preferred if o in opts] + sorted(others)
        return ordered
    return ['Mr','Mrs','Miss','Master','Rare']

df = load_data()
model = load_model()

st.title("Titanic Survival Predictor")
st.markdown("Interactive app: explore the Titanic dataset and predict whether a passenger would survive.")
page = st.sidebar.radio("Navigation", ["Home / Data", "Visualizations", "Predict", "Model Eval", "About"])

if page == "Home / Data":
    st.header("Dataset overview")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    with st.expander("Show raw data (first 50 rows)"):
        st.dataframe(df.head(50))
    st.subheader("Quick info")
    st.write(df.describe(include='all'))
    st.subheader("Missing values")
    miss = df.isnull().sum().sort_values(ascending=False)
    st.table(miss[miss > 0])
    st.subheader("Filter & inspect")
    cols_for_filter = ['Sex','Pclass','Embarked']
    filters = {}
    for c in cols_for_filter:
        if c in df.columns:
            options = sorted(df[c].dropna().unique())
            filters[c] = st.multiselect(f"Filter {c}", options, default=options)
    filtered = df.copy()
    for k,v in filters.items():
        filtered = filtered[filtered[k].isin(v)]
    st.write(f"Filtered rows: {filtered.shape[0]}")
    st.dataframe(filtered.head(100))

elif page == "Visualizations":
    st.header("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        if 'Sex' in df.columns and 'Survived' in df.columns:
            fig = px.histogram(df, x='Sex', color='Survived', barmode='group', histnorm='percent', title="Survival by Sex")
            st.plotly_chart(fig, use_container_width=True)
        if 'Pclass' in df.columns and 'Survived' in df.columns:
            fig2 = px.histogram(df, x='Pclass', color='Survived', barmode='group', histnorm='percent', title="Survival by Pclass")
            st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if 'Age' in df.columns:
            fig3 = px.histogram(df, x='Age', nbins=30, title="Age distribution")
            st.plotly_chart(fig3, use_container_width=True)
        if set(['Age','Fare','Survived']).issubset(df.columns):
            fig4 = px.scatter(df, x='Age', y='Fare', color='Survived', hover_data=['Name'] if 'Name' in df.columns else None, title="Fare vs Age")
            st.plotly_chart(fig4, use_container_width=True)

elif page == "Predict":
    st.header("Make a prediction")
    pclass = st.selectbox("Pclass", options=[1,2,3], index=1)
    sex = st.selectbox("Sex", options=['male','female'], index=0)
    age = st.slider("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=0.1)
    embarked = st.selectbox("Embarked", options=['S','C','Q'] if 'Embarked' in df.columns else ['S','C','Q'])
    title_opts = make_title_options(df)
    title = st.selectbox("Title", options=title_opts, index=0 if 'Mr' in title_opts else 0)
    if st.button("Predict"):
        input_df = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked,
            'Title': title
        }])
        if model is None:
            st.error("Model not loaded.")
        else:
            try:
                proba = model.predict_proba(input_df)[:,1][0] if hasattr(model, "predict_proba") else None
                pred = model.predict(input_df)[0]
                st.subheader("Result")
                st.write("Prediction:", "✅ Survived" if pred == 1 else "❌ Did not survive")
                if proba is not None:
                    st.write(f"Survival probability: {proba:.3f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.dataframe(input_df)

elif page == "Model Eval":
    st.header("Model evaluation")
    if model is None:
        st.error("Model not loaded.")
    else:
        if 'Survived' not in df.columns:
            st.warning("No 'Survived' column in dataset.")
        else:
            features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']
            X = df[features].copy()
            y = df['Survived'].copy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            try:
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                cr = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)
                st.metric("Accuracy", f"{acc:.3f}")
                st.dataframe(pd.DataFrame(cr).transpose())
                st.write("Confusion matrix:")
                st.write(cm.tolist())
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:,1]
                    auc = roc_auc_score(y_test, y_proba)
                    st.metric("ROC AUC", f"{auc:.3f}")
            except Exception as e:
                st.error(f"Evaluation error: {e}")

elif page == "About":
    st.header("About this app")
    st.markdown("""
    - Model: saved at `models/best_model.pkl`  
    - Data: loaded from GitHub raw URL  
    - Run locally: `streamlit run app.py`  
    """)
    st.markdown("Built for the Titanic ML project.")

st.markdown("---")
st.caption("Run this app from the project root folder.")
