import streamlit
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
import warnings
import datetime
from matplotlib.pyplot import figure
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pickle
from sklearn.metrics import roc_curve



st.write("""
# Loan Prediction using Logistic Regression 
  This App predicts the prepayment of clients
    """)

st.sidebar.header('User Input Features')

def user_input_features():
    CREDIT_SCORE = streamlit.number_input('Enter your credit score')
    st.write('The current credit score is ', CREDIT_SCORE)

    FIRST_TIME_HOMEBUYER_FLAG = streamlit.number_input('Is this your first home ?, 1 for yes and 0 for no')
    if FIRST_TIME_HOMEBUYER_FLAG == 1:
        st.write("It's your fist home")
    else:
        st.write('Not your first home')

    MORTGAGE_INSURANCE_PERCENTAGE = st.number_input("Enter the mortgage insurance percentage")
    st.write('Your mortgage insurance percentage is: {}'.format(MORTGAGE_INSURANCE_PERCENTAGE))

    NUMBER_OF_UNITS = st.number_input('Enter the number of units')
    st.write('Your number of units is: {}'.format(NUMBER_OF_UNITS))

    ORIGINAL_COMBINED_LOAN_TO_VALUE = st.number_input('Enter the original combined loan to value')
    st.write('Your number of original combined loan to value is: {}'.format(ORIGINAL_COMBINED_LOAN_TO_VALUE))

    ORIGINAL_DEBT_TO_INCOME_RATIO = st.number_input('Enter the original debt to income ratio')
    st.write('Your number of original debt to income ratio is: {}'.format(ORIGINAL_DEBT_TO_INCOME_RATIO))

    ORIGINAL_UPB = st.number_input('Enter the original UDB')
    st.write('Your number of original UDB is: {}'.format(ORIGINAL_UPB))

    ORIGINAL_LOAN_TO_VALUE = st.number_input('Enter the original loan to value')
    st.write('Your number of original loan to value is: {}'.format(ORIGINAL_LOAN_TO_VALUE))

    ORIGINAL_INTEREST_RATE = st.number_input('Enter the original interest rate')
    st.write('Your number of original interest rate is: {}'.format(ORIGINAL_INTEREST_RATE))

    PREPAYMENT_PENALTY_MORTGAGE_FLAG = st.number_input('Is there a penalty if you prepaid the mortgage?, 1 for yes and 0 for no')
    if PREPAYMENT_PENALTY_MORTGAGE_FLAG == 1:
        st.write("Yes there is a penalty")
    else:
        st.write('No there is not a penalty')

    ORIGINAL_LOAN_TERM = st.number_input('Enter the original loan term in years')
    st.write('Your original loan term in years is: {}'.format(ORIGINAL_LOAN_TERM))

    NUMBER_OF_BORROWERS = st.number_input('Enter the number of borrowers')
    st.write('The number of borrowers is: {}'.format(NUMBER_OF_BORROWERS))

    DELINQUENT =  st.number_input('Are you delinquent in paying your mortgage payment?'
                                  '1 for yes and 0 for no')
    if DELINQUENT == 1:
        st.write("Yes you are delinquent")
    else:
        st.write('No you are not delinquent')

    REPAY_DURATION = st.number_input('Enter your loan duration in months')
    st.write('Loan duration is: {}'.format(REPAY_DURATION))

    OCCUPANCY_STATUS = st.selectbox('Enter your occupancy status', options= ['Investment', 'Owner_Occupied',
                                                                             'Second_Home'])
    OCCUPANCY_STATUS_Investment = 0
    OCCUPANCY_STATUS_Owner_Occupied = 0
    OCCUPANCY_STATUS_Second_Home = 0

    if OCCUPANCY_STATUS == 'Investment':
        OCCUPANCY_STATUS_Investment = 1
        st.write('Your occupancy status is investment')
    elif OCCUPANCY_STATUS == 'Owner_Occupied':
        OCCUPANCY_STATUS_Owner_Occupied = 1
        st.write('Your occupancy status is owner occupied')
    else:
        OCCUPANCY_STATUS_Second_Home = 1
        st.write('Your occupancy status is second home')

    CHANNEL = st.selectbox('Enter your loan channel', options= ['Broker', 'Correspondent', 'Retail',
                                 'TPO Not specified'])
    CHANNEL_Broker = 0
    CHANNEL_Correspondent = 0
    CHANNEL_Retail = 0
    CHANNEL_TPO_Not_specified = 0

    if CHANNEL == 'Broker':
        CHANNEL_Broker = 1
        st.write('Your loan channel is Broker')
    elif CHANNEL == 'Correspondent':
        CHANNEL_Correspondent = 1
        st.write('Your loan channel is Correspondent')
    elif CHANNEL == 'Retail':
        CHANNEL_Retail = 1
        st.write('Your loan channel is Retail')
    else:
        CHANNEL_TPO_Not_specified = 1
        st.write('Your loan channel is TPO not specified')

    PRODUCT_TYPE_FRM = st.number_input('is your product type FRM or not, 1 for yes 0 for no')
    if PRODUCT_TYPE_FRM == 1:
        st.write('Your product type is FRM')
    else:
        st.write('Your product type is not FRM')

    PROPERTY_TYPE = st.selectbox('Enter your property type', options= ['Manufactured_Housing', 'Co_op', 'Condo',
                                 'PUD', 'Single_Family'])

    PROPERTY_TYPE__Manufactured_Housing = 0
    PROPERTY_TYPE_Co_op = 0
    PROPERTY_TYPE_Condo = 0
    PROPERTY_TYPE_PUD = 0
    PROPERTY_TYPE_Single_Family = 0

    if PROPERTY_TYPE == 'Manufactured_Housing':
        PROPERTY_TYPE__Manufactured_Housing = 1
        st.write('Your property type is Manufactured_Housing')
    elif PROPERTY_TYPE == 'Co_op':
        PROPERTY_TYPE_Co_op = 1
        st.write('Your property type is Co_op')
    elif PROPERTY_TYPE == 'Condo':
        PROPERTY_TYPE_Condo = 1
        st.write('Your property type is Condo')
    elif PROPERTY_TYPE == 'PUD':
        PROPERTY_TYPE_PUD = 1
        st.write('Your property type is PUD')
    else:
        PROPERTY_TYPE_Single_Family = 1
        st.write('Your property type is Single_Family')

    LOAN_PURPOSE = st.selectbox('Enter your property type', options= ['Purchase', 'Refinance_CashOut', 'Refinance_NoCashOut'])

    LOAN_PURPOSE_Purchase = 0
    LOAN_PURPOSE_Refinance_CashOut = 0
    LOAN_PURPOSE_Refinance_NoCashOut = 0

    if LOAN_PURPOSE == 'Purchase':
        LOAN_PURPOSE_Purchase = 1
        st.write('Your Loan Purpose is Purchase')
    elif LOAN_PURPOSE == 'Refinance_CashOut':
        LOAN_PURPOSE_Refinance_CashOut = 1
        st.write('Your Loan Purpose is Refinance_CashOut')
    else:
        LOAN_PURPOSE_Refinance_NoCashOut = 1
        st.write('Your Loan Purpose is Refinance_NoCashOut')

    data = {
        'CREDIT_SCORE': CREDIT_SCORE,
        'FIRST_TIME_HOMEBUYER_FLAG': FIRST_TIME_HOMEBUYER_FLAG,
        'MORTGAGE_INSURANCE_PERCENTAGE': MORTGAGE_INSURANCE_PERCENTAGE,
        'NUMBER_OF_UNITS': NUMBER_OF_UNITS,
        'ORIGINAL_COMBINED_LOAN_TO_VALUE': ORIGINAL_COMBINED_LOAN_TO_VALUE,
        'ORIGINAL_DEBT_TO_INCOME_RATIO': ORIGINAL_DEBT_TO_INCOME_RATIO,
        'ORIGINAL_UPB': ORIGINAL_UPB,
        'ORIGINAL_LOAN_TO_VALUE': ORIGINAL_LOAN_TO_VALUE,
        'ORIGINAL_INTEREST_RATE': ORIGINAL_INTEREST_RATE,
        'PREPAYMENT_PENALTY_MORTGAGE_FLAG': PREPAYMENT_PENALTY_MORTGAGE_FLAG,
        'ORIGINAL_LOAN_TERM': ORIGINAL_LOAN_TERM,
        'NUMBER_OF_BORROWERS': NUMBER_OF_BORROWERS,
        'DELINQUENT': DELINQUENT,
        'REPAY_DURATION': REPAY_DURATION,
        'OCCUPANCY_STATUS_Investment': OCCUPANCY_STATUS_Investment,
        'OCCUPANCY_STATUS_Owner_Occupied': OCCUPANCY_STATUS_Owner_Occupied,
        'OCCUPANCY_STATUS_Second_Home': OCCUPANCY_STATUS_Second_Home,
        'CHANNEL_Broker': CHANNEL_Broker,
        'CHANNEL_Correspondent': CHANNEL_Correspondent,
        'CHANNEL_Retail': CHANNEL_Retail,
        'CHANNEL_TPO_Not_specified': CHANNEL_TPO_Not_specified,
        'PRODUCT_TYPE_FRM': PRODUCT_TYPE_FRM,
        'PROPERTY_TYPE__Manufactured_Housing': PROPERTY_TYPE__Manufactured_Housing,
        'PROPERTY_TYPE_Co_op': PROPERTY_TYPE_Co_op,
        'PROPERTY_TYPE_Condo':PROPERTY_TYPE_Condo,
        'PROPERTY_TYPE_PUD': PROPERTY_TYPE_PUD,
        'PROPERTY_TYPE_Single_Family': PROPERTY_TYPE_Single_Family,
        'LOAN_PURPOSE_Purchase': LOAN_PURPOSE_Purchase,
        'LOAN_PURPOSE_Refinance_CashOut': LOAN_PURPOSE_Refinance_CashOut,
        'LOAN_PURPOSE_Refinance_NoCashOut': LOAN_PURPOSE_Refinance_NoCashOut

    }
    features = pd.DataFrame(data, index = [0])
    return features

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

#MODEL

loans_df = pd.read_csv("E:/Omar's Stuff/College related/Technocolab Internship/Materials/Dataset/loan_level_500k.csv")

# features_modifying
loans_df['FIRST_TIME_HOMEBUYER_FLAG'] = [1 if i == 'Y' else 0 if i == 'N'
                                        else np.nan for i in loans_df['FIRST_TIME_HOMEBUYER_FLAG']]

# we also need to change the FIRST_PAYMENT_DATE dtype to be datetime

loans_df['FIRST_PAYMENT_DATE'] = [datetime.datetime.strptime(str(i), '%Y%m') for i in loans_df['FIRST_PAYMENT_DATE']]

# we also need to change the MATURITY_DATE dtype to be datetime

loans_df['MATURITY_DATE'] = [datetime.datetime.strptime(str(i), '%Y%m') for i in loans_df['MATURITY_DATE']]

# we also need to change the POSTAL_CODE dtype to be string
loans_df['POSTAL_CODE'] = loans_df['POSTAL_CODE'].astype('str')

loans_df['PREPAYMENT_PENALTY_MORTGAGE_FLAG'] = [1 if i == 'Y' else 0 for i in
                                                loans_df['PREPAYMENT_PENALTY_MORTGAGE_FLAG']]

# loans_df['LTV_range'] = pd.cut(df.LTV,[0,25,50,1000],3,labels=['Low','Medium','High'])
loans_df['CreditRange'] = pd.cut(loans_df.CREDIT_SCORE, [550, 650, 700, 750, 1e6], 4, labels=[1, 2, 3, 4])

numerical_cols = np.array(loans_df.describe().columns)
categorical_cols = np.array(loans_df.describe(include=object).columns)

for col in numerical_cols:
    loans_df[col] = pd.to_numeric(loans_df[col])
    loans_df[col].fillna(loans_df[col].mean(), inplace=True)

for col in categorical_cols:
    loans_df[col].fillna(loans_df[col].mode().iloc[0], inplace=True)
loans_df['REPAY_DURATION'] = loans_df['MATURITY_DATE'].dt.year - loans_df['FIRST_PAYMENT_DATE'].dt.year

# Making the encoding more understandable
loans_df['PROPERTY_TYPE'] = ['Condo' if i == 'CO' else 'PUD' if i == 'PU' else ' Manufactured_Housing' if i == 'MH'
                            else 'Single_Family' if i == 'SF' else 'Co-op' for i in loans_df['PROPERTY_TYPE']]

# Making the encoding more understandable
loans_df['LOAN_PURPOSE'] = ['Purchase' if i == 'P' else 'Refinance_CashOut' if i == 'C' else 'Refinance_NoCashOut'
                            if i == 'N' else 'unknown' for i in loans_df['LOAN_PURPOSE']]


# Making the encoding more understandable
loans_df['OCCUPANCY_STATUS'] = ['Owner_Occupied' if i == 'O' else 'Investment' if i == 'I' else 'Second_Home'
                                for i in loans_df['OCCUPANCY_STATUS']]

# Making the encoding more understandable
loans_df['CHANNEL'] = ['Retail' if i == 'R' else 'Broker' if i == 'B' else 'Correspondent' if i == 'C'
                        else 'TPO Not specified' for i in loans_df['CHANNEL']]


X = loans_df.copy()
y = loans_df['PREPAID'].copy()

X.drop('PREPAID', inplace=True, axis=1)

X.drop(['POSTAL_CODE', 'SELLER_NAME'], inplace=True, axis=1)

X.drop('FIRST_PAYMENT_DATE', inplace=True, axis=1)

X.drop(['MATURITY_DATE', 'LOAN_SEQUENCE_NUMBER', 'METROPOLITAN_STATISTICAL_AREA'], inplace=True, axis=1)

X.drop(['PROPERTY_STATE', 'SERVICER_NAME'], inplace=True, axis=1)

X.drop('CreditRange', inplace=True, axis=1)

print(X.describe(include=object))

# ### Scalling and Creating dummy variables

cat_vars = X.describe(include=object).columns.tolist()

for var in cat_vars:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(X[var], prefix=var)
    data1 = X.join(cat_list)
    X = data1

data_vars = X.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

X_final = X[to_keep]

X_final.columns = [i.replace(" ", "_") for i in X_final.columns]
X_final.columns = [i.replace("-", "_") for i in X_final.columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=10, test_size=0.3)

logreg = LogisticRegression(fit_intercept=False)
logreg.fit(X_train, y_train)

logreg = LogisticRegression(fit_intercept=False)
logreg.fit(X_train, y_train)

prediction = logreg.predict(df)
prediction_proba = logreg.predict_proba(df)

score = logreg.score(X_test, y_test)
st.subheader('Accuracy of the model')
st.write('Accuracy is: {}'.format(score * 100))

st.subheader('Prediction')
if prediction:
    st.write("The customer will prepay")
else:
    st.write("The customer will not prepay")