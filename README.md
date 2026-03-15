# DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction
A machine learning project that analyzes customer support interactions and predicts Customer Satisfaction (CSAT) scores using supervised learning, data analytics, and NLP techniques — all deployed through an interactive Streamlit application.

# Project Overview
This project addresses an important business challenge in e-commerce customer support: understanding and predicting customer satisfaction based on service interactions.

# The system analyzes support data such as:
1.communication channel
2.issue category
3.handling time
4.product information
5.customer remarks
6.and predicts the CSAT Score to help companies improve service quality and customer experience.


## 🧠 Models Used

| Model   | Purpose                         | Algorithm                          |
|---------|---------------------------------|------------------------------------|
| Model 1 | 	Baseline CSAT prediction     | Linear Regression                  |
| Model 2 | Improve prediction accuracy     | Random Forest Regressor            |
| Model 3 | Final optimized model	Gradient| Boosting Regressor 





# Dataset

The dataset contains customer service interaction records including:
channel_name,
category,
sub_category,
customer remarks,
item price,
handling time,
agent information,
CSAT Score

## How to Run This Project
1. Clone this repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
  streamlit run app.py 

## Projects Highlights
✔ Customer satisfaction prediction using Machine Learning
✔ Exploratory Data Analysis with 15+ visualizations
✔ Feature engineering and hypothesis testing
✔ Hyperparameter tuning with RandomizedSearchCV
✔ Model deployment through Streamlit web application
✔ Real-time CSAT score prediction
