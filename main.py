import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import base64
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.write('''# *Client Churn Predictor*''')
  
uploaded_file = st.file_uploader("Upload a CSV file data test", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write(
    '''
    ### Input Data ({} Customers)
    '''.format(input_df.shape[0])
    )
    st.dataframe(input_df)
    st.write('')
    xgb = pickle.load(open( "churn_model.sav", "rb" ) )

    X = input_df.drop(labels = ['id'], axis = 1)

    X['total_calls'] = X['total_day_calls'] + X['total_eve_calls'] + X['total_night_calls'] + X['total_intl_calls']
    X['total_charge'] = X['total_day_charge'] + X['total_eve_charge'] + X['total_night_charge'] + X['total_night_charge']
    X['day_calls_per_charge'] = X['total_day_calls'] / X['total_day_charge']
    X['eve_calls_per_charge'] = X['total_eve_calls'] / X['total_eve_charge']
    X['night_calls_per_charge'] = X['total_night_calls'] / X['total_night_charge']
    X['intl_calls_per_charge'] = X['total_intl_calls'] / X['total_intl_charge']
    X = X.fillna(0)
    def service_calls_group(row):
        if row['number_customer_service_calls'] == 0:
            return 'No Service Calls'
        elif row['number_customer_service_calls'] >= 1 and row['number_customer_service_calls'] <= 3:
            return 'Few Service Calls'
        else:
            return 'Many Service Calls'

    X['service_calls_group'] = X.apply(service_calls_group, axis=1)
    le = LabelEncoder()
    X['state'] = le.fit_transform(X['state'])
    X = pd.get_dummies(X)

    threshold = .45
    y_preds = xgb.predict(X)
    predicted_proba = xgb.predict_proba(X)
    y_preds = (predicted_proba [:,1] >= threshold).astype('int')

    y_preds_series = pd.Series(y_preds)
    value_counts = y_preds_series.value_counts()
    st.write(
    '''
    ### Total {} Customers Churn and {} Customer Not Churn
    '''.format(value_counts.iloc[1], value_counts.iloc[0])
    )
    #check proportion of Churn
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(value_counts, autopct='%.2f',
        explode=[0.1,0],
        labels=["No","Yes"],
        textprops={'fontsize': 14},
        colors=["gray","red"], 
        startangle=35)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    # Display the pie chart in the Streamlit app
    st.pyplot(fig)
    y_preds_series = y_preds_series.replace({0: "Not Churn", 1: "Churn"})
    csv = pd.concat([input_df['id'],y_preds_series],axis=1)
    csv = csv.rename(columns={0: 'Status'}).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.write('''''')
    st.write('''''')
    st.write('''### **⬇️ Download At-Risk Customer Id's**''')
    href = f'<a href="data:file/csv;base64,{b64}" download="at_risk_customerids.csv">Download csv file</a>'
    st.write(href, unsafe_allow_html=True)