# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing
from scipy.stats import norm


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

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
nrmlz_df = pd.DataFrame(d, columns=column_names)
print(nrmlz_df.head())

