import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns

# Load saved model and data
model = pickle.load(open('titanic_model.pkl', 'rb'))
df = pd.read_csv('cleaned_titanic.csv')

st.set_page_config(page_title="Titanic Survival Dashboard", layout="wide")

st.title("🚢 Titanic Survival Dashboard")

tab1, tab2 = st.tabs(["🔮 Survival Predictor", "📊 Advanced Analytics"])

with tab1:
    st.header("Predict Your Survival")
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Ticket Class (1=Best)", [1, 2, 3])
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 0, 80, 25)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)

    if st.button("Predict Survival"):
        sex_val = 0 if sex == 'male' else 1
        # Feature vector for prediction
        features = [[pclass, sex_val, age, sibsp, parch, fare, 0]]
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("✅ You likely would have SURVIVED.")
        else:
            st.error("❌ You likely would NOT have survived.")

with tab2:
    st.header("Advanced Dataset Insights")

    # ADVANCED FEATURE 1: Family Size Analysis
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1
    fig_fam = px.histogram(df, x="FamilySize", color="survived", barmode="group",
                           title="Survival Rate by Family Size")
    st.plotly_chart(fig_fam)

    # ADVANCED FEATURE 2: Fare vs. Class vs. Survival
    fig_fare = px.scatter(df, x="age", y="fare", color="survived", size="pclass",
                          hover_data=['pclass'], title="Correlation: Age, Fare, and Survival")
    st.plotly_chart(fig_fare)
