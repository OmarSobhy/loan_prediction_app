# importing libraries
import pandas as pd
import numpy as np
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

warnings.filterwarnings("ignore")

# ### Data Exploring Phase
# Loading data
loans_df = pd.read_csv("E:/Omar's Stuff/College related/Technocolab Internship/Materials/Dataset/loan_level_500k.csv")

loans_df.head(10)

loans_df.info()
# It looks like there is a lot of data type inconsistencies on the data like for example the FIRST_PAYMENT_DATE dtype
# should be 'datetime' not 'int'.

print("the shape of the dataset is: {}".format(loans_df.shape))

# the dataset has 27 features or columns of data.

print(loans_df.describe())

print(loans_df.duplicated().sum())

print(loans_df.describe(include=object))

print(loans_df.isnull().sum())

# the data has 14 'numeric' columns and 11 'categorical' columns

# visualizing missing data

# Visualizing Null Values
sns.heatmap(loans_df.isnull(), cbar=False, cmap='viridis')
plt.show()

# from the previous heatmap its shown that the top 3 columns that have missing data are FIRST_TIME_HOMEBUYER_FLAG,
# METROPOLITAN_STATISTICAL_AREA and MORTGAGE_INSURANCE_PERCENTAGE and we can't drop them as they may contain important
# data to our analysis

# let's change the type of FIRST_TIME_HOMEBUYER_FLAG to boolean
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
loans_df.CreditRange.unique()

# lets split the data to categorical and numerical data
numerical_cols = np.array(loans_df.describe().columns)
categorical_cols = np.array(loans_df.describe(include=object).columns)

for col in numerical_cols:
    loans_df[col] = pd.to_numeric(loans_df[col])
    loans_df[col].fillna(loans_df[col].mean(), inplace=True)

print(loans_df.head())

# Visualizing Null Values
sns.heatmap(loans_df.isnull(), cbar=False, cmap='viridis')
plt.show()

# from the previous heatmap it shows that we filled most of the numerical null values with the mean and the problem
# remains in the 'FIRST_TIME_HOMEBUYER_FLAG' column as I still need to investigate the means to replace the null values 
# in such columns.

# I decided to replace the missing data in categorical data with the mode just for now 
# , but this matter will need further investigation
for col in categorical_cols:
    loans_df[col].fillna(loans_df[col].mode().iloc[0], inplace=True)

# Visualizing Null Values
sns.heatmap(loans_df.isnull(), cbar=False, cmap='viridis')
plt.show()
# Now it's all clear, I hope.

# creating a loan duration column
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

print(loans_df.groupby(['LOAN_PURPOSE']).PREPAID.value_counts(normalize=True))

# All three loan types have similar 'PREPAID' distributions and it seems that
# it does not have that huge effect on the customer being a PREPAID or not 
# with a small difference that the 'PURCHASE' type has a higher 'PREPAID' ratio than the other two types.

print(loans_df.groupby(['OCCUPANCY_STATUS']).PREPAID.value_counts(normalize=True))

# The 'Investment' type looks like to have the smallest ratio on customer being 'PREPAID'
# which means that it's less likely that if the OCCUPANCY_STATUS is Investment
# that the customer will prepay the loan that the other two types of OCCUPANCY_STATUS.

print(loans_df.groupby(['OCCUPANCY_STATUS'])['CREDIT_SCORE'].describe())

# ### Visualizations

sns.boxplot(data=loans_df, y='CREDIT_SCORE', x='PREPAID')
plt.show()
# the mean of the credit score is slightly higher for prepaid customers

sns.histplot(x=loans_df['CREDIT_SCORE'], hue=loans_df['PREPAID'], bins=40, kde=True)
plt.title('Credit Score Distribution between Prepaid and Non-Prepaid Customers.')
plt.show()

sns.histplot(x=loans_df['CREDIT_SCORE'], hue=loans_df['LOAN_PURPOSE'], stat='probability', bins=30)
plt.show()

sns.catplot(x="LOAN_PURPOSE", y="CREDIT_SCORE", hue="PREPAID", kind="box", data=loans_df)
plt.xticks(rotation=90)
plt.show()

sns.catplot(x="OCCUPANCY_STATUS", y="CREDIT_SCORE", hue="PREPAID", kind="box", data=loans_df)
plt.xticks(rotation=90)
plt.show()

sns.countplot(x='OCCUPANCY_STATUS', hue='PREPAID', data=loans_df)
plt.show()

figure(figsize=(20, 15), dpi=80)

sns.heatmap(loans_df.corr(), annot=True)
plt.show()
# from this plot this seems like there are some positive correlations like the one between
# 'ORIGINAL_COMBINED_LOAN_TO_VALUE' and 'MORTGAGE_INSURANCE_PERCENTAGE'
# And some negative correlations like the one between 'ORIGINAL_LOAN_TO_VALUE' and 'CREDIT_SCORE'
# And the best positive correlation found with 'PREPAID' was with 'ORIGINAL_UPB'

sns.scatterplot(x='MORTGAGE_INSURANCE_PERCENTAGE', y='ORIGINAL_COMBINED_LOAN_TO_VALUE',
                data=loans_df, hue='PREPAID', ci='sd')
plt.title('Correation between MORTGAGE_INSURANCE_PERCENTAGE and ORIGINAL_COMBINED_LOAN_TO_VALUE')
plt.show()

sns.scatterplot(x='MORTGAGE_INSURANCE_PERCENTAGE', y='ORIGINAL_LOAN_TO_VALUE',
                data=loans_df, hue='PREPAID', ci='sd')
plt.title('Correation between MORTGAGE_INSURANCE_PERCENTAGE and ORIGINAL_LOAN_TO_VALUE')

plt.show()

sns.relplot(
    data=loans_df, x="MORTGAGE_INSURANCE_PERCENTAGE", y="ORIGINAL_LOAN_TO_VALUE",
    col="PREPAID", hue="OCCUPANCY_STATUS", kind="scatter")
plt.show()

# ## Model Making

print(loans_df.info())

X = loans_df.copy()
y = loans_df['PREPAID'].copy()

X.drop('PREPAID', inplace=True, axis=1)

X.drop(['POSTAL_CODE', 'SELLER_NAME'], inplace=True, axis=1)

X.drop('FIRST_PAYMENT_DATE', inplace=True, axis=1)

X.drop(['MATURITY_DATE', 'LOAN_SEQUENCE_NUMBER', 'METROPOLITAN_STATISTICAL_AREA'], inplace=True, axis=1)

len(X.columns)

print(X.info())

X.describe(include=object)

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

X_final.info()
X_final.columns = [i.replace(" ", "_") for i in X_final.columns]
X_final.columns = [i.replace("-", "_") for i in X_final.columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# ### Feature Selection


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=10, test_size=0.3)

logreg = LogisticRegression(fit_intercept=False)
logreg.fit(X_train, y_train)

#predictors = X_train
#selector = RFE(logreg, n_features_to_select=1)
#selector = selector.fit(predictors, y_train)

#order = selector.ranking_

#feature_ranks = []
#for i in order:
 #   if i < 30:
  #      feature_ranks.append(f"{X_final.columns[i]}")

#columns_to_use = feature_ranks[:20]

# # Logistic Regression

logreg = LogisticRegression(fit_intercept=False)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

score = logreg.score(X_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# False positive Rate,True positive Rate
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='lg')
plt.xlabel('fpr', fontsize=20)
plt.ylabel('tpr', fontsize=20)
plt.title('Logistic Regression ROC curve', fontsize=20)
plt.show()

# We got a better accuracy with the selected columns,
# I will change the number of selected columns and see what is going to be the new accuracy

# The best accuracy I got is when I used 21 features from the selected columns

#pickle.dump(logreg,
#           open("E:\Omar's Stuff\College related\Technocolab Internship\ML Model Deployment\Logistic Regression.pkl",
#               'wb'))

print("X columns are :{}".format(X_final.columns))