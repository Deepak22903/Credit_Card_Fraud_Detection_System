import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from termcolor import colored as cl  # text customization
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6, 3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

from pylab import rcParams
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('creditcard.csv')

fraud_cases = len(data[data['Class'] == 1])
print(' Number of Fraud Cases:', fraud_cases)

non_fraud_cases = len(data[data['Class'] == 0])
print('Number of Non Fraud Cases:', non_fraud_cases)

fraud = data[data['Class'] == 1]
genuine = data[data['Class'] == 0]

fraud.Amount.describe()
genuine.Amount.describe()

data.hist(figsize=(25, 20), color='yellow')
plt.savefig('analysis.png')
plt.show()
plt.clf()

rcParams['figure.figsize'] = 16, 8
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time, genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
plt.savefig('Fraud_or_genuine.png')
plt.clf()

data = pd.read_csv("creditcard.csv")

Total_transactions = len(data)
normal = len(data[data.Class == 0])
fraudulent = len(data[data.Class == 1])
fraud_percentage = round(fraudulent / normal * 100, 2)

data.info()
sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

data.drop(['Time'], axis=1, inplace=True)

data.drop_duplicates(inplace=True)

X = data.drop('Class', axis=1).values
y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

RF = RandomForestClassifier(max_depth=4, criterion='entropy')
RF.fit(X_train, y_train)
dt_yhat = RF.predict(X_test)

print('Accuracy score of the Classifier is {}'.format(accuracy_score(y_test, dt_yhat)))
