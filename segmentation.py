import streamlit as st 
import pandas as pd
import numpy as np
import joblib


kmeans = joblib.load('kmeans_model.pkl')
sc = joblib.load('scaler.pkl')

st.title("💼 Customer Segmentation App")
st.write("Enter customer details to find out which cluster they belong to.")


age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income ($)", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=500)
num_store = st.number_input("Number of Store Purchases", min_value=0, max_value=50, value=5)
num_web = st.number_input("Number of Web Purchases", min_value=0, max_value=50, value=3)
num_visits = st.number_input("Monthly Web Visits", min_value=0, max_value=50, value=4)
recency = st.number_input("Days Since Last Purchase (Recency)", min_value=0, max_value=365, value=30)

if st.button("🔍 Predict Cluster"):
    data = pd.DataFrame([[age, income, total_spending, num_store, num_web, num_visits, recency]],
                        columns=["Age", "Income", "Total_Spending", "NumStorePurchases", 
                                 "NumWebPurchases", "NumWebVisitsMonth", "Recency"])
    
    data_scaled = sc.transform(data)

    cluster = kmeans.predict(data_scaled)[0]

    st.success(f"The customer belongs to **Cluster {cluster}**.")

    if cluster == 0:
        st.info("💰 Wealthy but inactive — might need re-engagement campaigns.")
    elif cluster == 1:
        st.info("📉 Low-income and inactive — may respond to discounts and promotions.")
    elif cluster == 2:
        st.info("🌟 Loyal and high-spending customer — great for loyalty programs.")
    elif cluster == 3:
        st.info("🛒 Online-active but recently disengaged — consider personalized offers.")

st.caption("Model: KMeans | Data scaled before clustering.")
