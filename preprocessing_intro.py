import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing
from scipy.stats import norm


df = pd.read_csv("Concrete_Data.csv")
raw_data = df.to_numpy()
print('raw_data', raw_data.shape)

X = raw_data[:,0:(raw_data.shape[1] - 1)]
y = raw_data[:, (raw_data.shape[1] - 1)].reshape(-1, 1)
n, d = X.shape

X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]

print(y_df.describe())
print(X_df.describe())
pd.options.display.max_columns = 9
fig_n_per_row = 3

fig, axs = plt.subplots(math.ceil(d/fig_n_per_row), fig_n_per_row, figsize=(15, 15))

sns.kdeplot(data=y_df, ax=axs[2][2], color='green')
for _ in range(d):
    sns.kdeplot(data=df.iloc[:,_], ax=axs[math.floor(_/fig_n_per_row)][_%fig_n_per_row])

fig.tight_layout()
plt.show()

fig, axs = plt.subplots(math.ceil(d/fig_n_per_row), fig_n_per_row, figsize=(15, 15))
sns.distplot(y_df, ax=axs[2][2], fit=norm, kde=False, color='green')
for _ in range(d):
    sns.distplot(df.iloc[:,_], ax=axs[math.floor(_/fig_n_per_row)][_%fig_n_per_row], fit=norm, kde=False)

fig.tight_layout()
plt.show()

nrmlz = preprocessing.MinMaxScaler()
column_names = df.columns
d = nrmlz.fit_transform(df)
df_nrmlz = pd.DataFrame(d, columns=column_names)
print(df_nrmlz.head())

raw_data_nrmlz = df_nrmlz.to_numpy()
print('raw_data_nrmlz', raw_data_nrmlz.shape)

X_nrmlz = raw_data_nrmlz[:,0:(raw_data_nrmlz.shape[1] - 1)]
y_nrmlz = raw_data_nrmlz[:, (raw_data_nrmlz.shape[1] - 1)].reshape(-1, 1)
n, d = X_nrmlz.shape

X_df_nrmlz = df_nrmlz.iloc[:,:-1]
y_df_nrmlz = df_nrmlz.iloc[:,-1]

print(y_df_nrmlz.describe())
print(X_df_nrmlz.describe())
pd.options.display.max_columns = 9
fig_n_per_row = 3

fig, axs = plt.subplots(math.ceil(d/fig_n_per_row), fig_n_per_row, figsize=(15, 15))

sns.kdeplot(data=y_df_nrmlz, ax=axs[2][2], color='green')
for _ in range(d):
    sns.kdeplot(data=df_nrmlz.iloc[:,_], ax=axs[math.floor(_/fig_n_per_row)][_%fig_n_per_row])

fig.tight_layout()
plt.show()

fig, axs = plt.subplots(math.ceil(d/fig_n_per_row), fig_n_per_row, figsize=(15, 15))
sns.distplot(y_df_nrmlz, ax=axs[2][2], fit=norm, kde=False, color='green')
for _ in range(d):
    sns.distplot(df_nrmlz.iloc[:,_], ax=axs[math.floor(_/fig_n_per_row)][_%fig_n_per_row], fit=norm, kde=False)

fig.tight_layout()
plt.show()

