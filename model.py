
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# %% [code]
item_cat = pd.read_csv('item_categories.csv')
items = pd.read_csv('items.csv')
sales_train = pd.read_csv('sales_train.csv')
sample_submission = pd.read_csv('sample_submission.csv')
shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')

# %% [code]
tr1 = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price':'mean','item_cnt_day':'sum'})
tr1.reset_index(inplace=True)
tr1.rename({'item_price':'Avg_item_price','item_cnt_day':'item_cnt_month'},axis=1,inplace=True)


# %% [code]
tr1['item_cnt_month']=tr1['item_cnt_month'].clip(0,30)



tp = tr1.pivot_table(index=['shop_id' , 'item_id'],columns='date_block_num',values='item_cnt_month')

tp.replace(np.NaN,0,inplace=True)

tp.reset_index(inplace=True)


# %% [code]
# df1 = sales_train[['shop_id','item_id']]
# df1['obs'] = 1
# df1 = df1.groupby(['shop_id','item_id']).agg({'obs':'sum'}).reset_index()
# df1.shape
# df1.sort_values(by=['obs'],ascending = False, inplace=True)
# df1.head()
# df1.tail()

# %% [code]
# df1['obs'].describe()

# %% [code]
from sklearn.model_selection import train_test_split


# %% [code]
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# %% [code]
# test.shape
# tp.shape
test_df = pd.merge(test,tp,on=['shop_id','item_id'],how='left')


# %% [code]
test_df.replace(np.NaN,0,inplace=True)
test_df

# %% [code]
test_df

# %% [code]
# tp1 = pd.concat([tp,test_df.drop(['ID'],axis=1)],ignore_index=True)
# tp1.shape
# tp1.drop_duplicates(inplace=True)
# tp1.shape
# tp1

# %% [code]
X_test = np.array(test_df.iloc[:,-34:-1])
y_test = np.array(test_df.iloc[:,-1])


# %% [code]
X = tp.iloc[:,2:-1].values
y = tp.iloc[:,-1].values

# %% [code]
# X.shape
# y.shape

# %% [code]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=1, shuffle=True)

# %% [code]
from sklearn.linear_model import LinearRegression
from sklearn import metrics

model = LinearRegression()
rmse = []
r2 = []

# %% [code]
for train_index, test_index in kf.split(X):
#     print(train_index,test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train,y_train)
    prediction =  model.predict(X_test)
    rmse.append(np.sqrt(metrics.mean_squared_error(y_test,prediction)))
    r2.append(metrics.r2_score(y_test,prediction))

# %% [code]
import pickle
filename = 'lr_model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(rmse)
print(r2)

# %% [code]
x_test = np.array(test_df.iloc[:,-33:])
x_test.shape

# %% [code]
predictions1 = model.predict(x_test)
p1 = predictions1.round()

# %% [code]
p1 = p1.clip(0,30)


# %% [code]
sample_submission['item_cnt_month'] = p1

# %% [code]
sample_submission

# %% [code]
sample_submission.to_csv('sample_submission.csv', index=False)
