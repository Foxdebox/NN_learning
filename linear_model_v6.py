import pandas as pd
import numpy as np
df = pd.read_csv('train.csv')
features = ['OverallQual','GrLivArea','TotalBsmtSF']
label = ['SalePrice']
selected_df = df[features + label]
clean_df = selected_df.dropna()
x = clean_df[features].values
y_actual = clean_df[label].values
y_log = np.log1p(y_actual)
def normalize(data,means,stds):
    return (data-means)/stds
def train(x,y_log,lr,epochs):
    means = np.mean(x,axis=0)
    stds = np.std(x,axis=0)
    x_norm = normalize(x,means,stds)
    n = x_norm.shape[0]
    n_features = x_norm.shape[1]
    w = np.zeros((n_features,1))
    b = 0
    for epoch in range(epochs):
        y_predict = np.dot(x_norm,w) + b
        residual = y_predict - y_log
        loss = np.mean(residual**2)
        gradient_w = 2/n * np.dot(x_norm.T,residual)
        gradient_b = 2/n * np.sum(residual)
        w = w - gradient_w * lr
        b = b - gradient_b * lr
        if epoch % 100 == 0:
            print(f"第{epoch}轮，loss={loss:.5f}")
    return w,b,loss,means,stds
w,b,loss,means,stds = train(x,y_log,0.01,1000)
df_test = pd.read_csv('test.csv')
df_test_means = df_test[features].mean()
df_test[features] = df_test[features].fillna(df_test_means)
x_test = df_test[features].values
x_test_norm = normalize(x_test,means,stds)
y_predict_log = np.dot(x_test_norm,w) + b
y_predict = np.expm1(y_predict_log)
submission_df = pd.DataFrame()
submission_df['Id'] = df_test['Id']
submission_df['SalePrice'] = y_predict
submission_df.to_csv('submission.csv',index=False)
