import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
ohe = OneHotEncoder()
encoded = pd.DataFrame(ohe.fit_transform(data[['Fuel Type']]).toarray())
data = data.join(encoded)
data.rename('')

y = data['CO2 Emissions (g/km)'].copy()
X = data.drop('CO2 Emissions (g/km)', axis=1)

X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2)

