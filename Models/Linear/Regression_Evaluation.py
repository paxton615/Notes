#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pwd


# In[4]:


data = pd.read_csv('/Users/paxton615/GA/resource-datasets/football_combine/combine_train.csv')
combine = pd.read_csv('/Users/paxton615/GA/resource-datasets/football_combine/combine_train.csv')


# In[9]:


print(combine.dtypes)
print('-----------------------------------------')
print(combine.shape)
print('-----------------------------------------')
print(combine.head())
print('-----------------------------------------')
print(np.sum(combine.isnull()))
print('-----------------------------------------')
combine.describe()


# In[15]:


def correlation_heat_map(df):
    corrs = df.corr()

    # Set the default matplotlib figure size:
    fig, ax = plt.subplots(figsize=(16, 8))

    # Generate a mask for the upper triangle (taken from seaborn example gallery)
    mask = np.zeros_like(corrs, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Plot the heatmap with seaborn.
    # Assign the matplotlib axis the function returns. This will let us resize the labels.
    ax = sns.heatmap(corrs, mask=mask, annot=True)

    # Resize the labels.
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=45)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)

    # If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
    plt.show()


# In[36]:


correlation_heat_map(combine) 
combine.hist(figsize=(12,12))
combine.plot(kind='box', subplots=True, layout=(6,6), figsize=(12,12))
plt.show()


# #### focus on the position Wide Receiver

# In[39]:


combine.Position.value_counts().head()


# In[51]:


df = combine[combine['Position']== 'WR']

weight = df[['Weight']] # 注意这里的取值方式
height = df.HeightInchesTotal


# - modeling with slr and statsmodes

# In[285]:


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(weight, height)
print('intercept:', slr.intercept_)
print('coef:', slr.coef_)


# In[286]:


import statsmodels.api as sm

st_model = sm.OLS(height, sm.add_constant(weight)) # 不同点，先输入y，还要给x加上intercept
result = st_model.fit()
result.params


# In[287]:


summary = result.summary()
summary


# ### MSE

# In[288]:


from sklearn.metrics import mean_squared_error


# In[289]:


height_pred = slr.predict(weight)

slr_mse = mean_squared_error(height, height_pred)
base_mse = mean_squared_error(height, np.repeat(np.mean(height), len(height)))
print('slr_mse:', slr_mse)
print("base_mse:", base_mse)


# In[290]:


fig = plt.figure(figsize=(6,6))
plt.scatter(weight, height, label='true y')
plt.plot(weight, height_pred, c='darkred',alpha=0.75, label='regression line')
plt.plot(weight, np.repeat(np.mean(height), len(height)), 'y', label='baseline')
plt.xlabel('weight', fontsize=12)
plt.ylabel('height',fontsize=12, rotation=0)
plt.legend()
plt.show()


# In[133]:


fig = plt.figure(figsize=(6,6))
ax = fig.gca()
ax.scatter(weight, height, label='true y', c='steelblue' )
ax.plot(weight, height_pred, c='darkred',alpha=0.75, label='regression line')
ax.plot(weight, np.repeat(np.mean(height), len(height)), 'y', label='baseline')
ax.set_xlabel('weight', fontsize=12)
ax.set_ylabel('height',fontsize=12, rotation=0)
ax.tick_params(axis='both', labelsize=20)
plt.legend(loc='lower right')
plt.show()


# ### R2

# In[291]:


from sklearn.metrics import r2_score


# In[292]:


r2 = r2_score(height, height_pred)
print('r2:', r2)


# ### Remove outliers
# 根据正态分布，显著水平alpha=0.05的为检出水平，这样的outlier需特别留心。而alpha=0.01的，即可剔除。

# In[260]:


hnw = pd.DataFrame({'weight':weight.Weight, 'height':height})
h_mean = hnw.height.mean()
h_std = hnw.height.std()
hnw['outlier'] = (np.abs(hnw.height - h_mean) > 1.5 * h_std)
len(hnw[hnw['outlier']==True])


# In[254]:


# hnw['outlier_3'] = (np.abs(hnw.height - h_mean) > 2 * h_std)
# len(hnw[hnw['outlier_3']==True])


# In[262]:


hnw = hnw[hnw['outlier']==False]


# In[263]:


hnw


# In[228]:


lr = LinearRegression()
lr.fit(hnw.weight.to_frame(), hnw.height)
print(lr.intercept_)
print(lr.coef_[0])


# In[231]:


no_outlier_pred = lr.predict(hnw.weight.to_frame())


# In[242]:


fig = plt.figure(figsize=(6,6))
ax = plt.gca()

ax.scatter(weight, height, label='true y', c='steelblue' )
ax.plot([weight.min(), weight.max()],[ no_outlier_pred.min(), no_outlier_pred.max()], 'r--', label='reg_no_outlier')
ax.plot([weight.min(), weight.max()],[height_pred.min(), height_pred.max()], c='g', label='reg model')
plt.legend()
ax.set_xlabel('weight')
ax.set_ylabel('height')
plt.show()


# ### 上图很明显功能看出，祛除了outlier的，竟然表现的不好。这点让我有点意外。

# In[264]:


print('r2_reg_model:',r2_score(height, height_pred) )
print('r2_reg_no_outlier:',r2_score(hnw.height, no_outlier_pred) )


# - r2 scores said the same thing, no_outlier model is worse

# #### Examing residuals

# In[266]:


(height-height_pred).hist(bins=50);


# In[267]:


(hnw.height - no_outlier_pred).hist(bins=50);


# ### Get the $R^2$ value for your original model predicting values from the test data

# In[269]:


combine_test = pd.read_csv('/Users/paxton615/GA/resource-datasets/football_combine/combine_test.csv')


# In[270]:


# combine_test.shape


# In[273]:


# combine_test.isnull().sum()


# In[274]:


wr_test = combine_test[combine_test['Position']=='WR']


# In[293]:


test_weight = wr_test[['Weight']]


# In[294]:


test_height = wr_test['HeightInchesTotal']


# In[295]:


test_pred = slr.predict(test_weight)


# In[296]:


test_r2 = r2_score(test_height, test_pred)


# In[299]:


print('r2_test:', test_r2)
print('r2_train:', r2)


# In[301]:


print('test_mse:', mean_squared_error(test_height, test_pred))
print('train_mse:', mean_squared_error(height, height_pred))

