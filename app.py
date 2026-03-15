import streamlit as st
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("eCommerce_Customer_support_data.csv")

# Load trained model
model = joblib.load("csat_prediction_model.pkl")

st.set_page_config(page_title="AI Customer Satisfaction Predictor", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🤖 CSAT Prediction", "📊 Customer Insights"]
)

# ================= HOME PAGE =================
if page == "🏠 Home":

    st.title("🛍 AI Customer Satisfaction Prediction System")

    st.write(
        """
        This AI-powered application helps e-commerce companies **predict customer satisfaction (CSAT)** 
        based on customer service interactions.
        
        It uses Machine Learning to analyze:
        - Customer support channels
        - Product categories
        - Handling time
        - Item price
        - Customer location
        """
    )

    st.subheader("🔎 Explore Dataset")

    st.dataframe(df.head())

    st.subheader("📈 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Features", df.shape[1])
    col3.metric("Unique Cities", df["Customer_City"].nunique())

    st.write("### Featured Insights")

    st.bar_chart(df["channel_name"].value_counts())


# ================= CSAT PREDICTION =================
elif page == "🤖 CSAT Prediction":

    st.title("🤖 Predict Customer Satisfaction Score")

    st.write("Enter customer support details to predict CSAT score.")

    channel = st.selectbox("Customer Support Channel", df["channel_name"].unique())
    category = st.selectbox("Product Category", df["Product_category"].unique())
    price = st.number_input("Item Price", 0, 100000)
    handling_time = st.number_input("Handling Time (minutes)", 0, 200)

    city = st.selectbox("Customer City", df["Customer_City"].unique())

    if st.button("Predict CSAT Score"):

        input_data = pd.DataFrame({
            "channel_name":[channel],
            "Product_category":[category],
            "Item_price":[price],
            "connected_handling_time":[handling_time],
            "Customer_City":[city]
        })

        # Convert categorical columns
        input_data = pd.get_dummies(input_data)

        # Align columns with training data
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_data)

        st.success(f"Predicted CSAT Score: ⭐ {round(prediction[0],2)}")


# ================= CUSTOMER INSIGHTS =================
elif page == "📊 Customer Insights":

    st.title("📊 Customer Support Insights Dashboard")

    st.subheader("Channel Distribution")

    st.bar_chart(df["channel_name"].value_counts())

    st.subheader("Product Category Distribution")

    st.bar_chart(df["Product_category"].value_counts())

    st.subheader("Average CSAT by Channel")

    csat_channel = df.groupby("channel_name")["CSAT Score"].mean()

    st.bar_chart(csat_channel)

    st.subheader("Handling Time vs CSAT")

    st.scatter_chart(df[["connected_handling_time","CSAT Score"]])

    st.write("These insights help companies identify service issues and improve customer satisfaction.")
