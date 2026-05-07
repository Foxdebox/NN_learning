import numpy as np
raw_data = [[50, 20, 10, 200],
    [80, 10, 5, 450],
    [120, 5, 2, 800],
    [30, 30, 15, 80],
    [200, 2, 1, 1500],
    [60, 15, 8, 300],
    [100, 8, 4, 600],
    [40, 25, 12, 120],
    [150, 3, 3, 1100],
    [90, 12, 6, 500]]
data = np.array(raw_data)
x = data[:,:3]
y_actual = data[:,-1:]
def normalize(data,means,stds):
  return (data-means)/stds
w = np.zeros((3,1))
b = 0
def train(x,y_actual,lr,epochs):
  means = np.mean(x,axis=0)
  stds = np.std(x,axis=0)
  x_norm = normalize(x,means,stds)
  n= x_norm.shape[0]
  n_features = x_norm.shape[1]
  w = np.zeros((n_features,1))
  b = 0
  for epoch in range(epochs):
    y_predict = np.dot(x_norm,w) + b
    residual = y_predict - y_actual
    loss = np.mean(residual**2)
    gradient_w = 2/n * np.dot(x_norm.T,residual)
    gradient_b = 2/n * np.mean(residual)
    w = w - gradient_w * lr
    b = b - gradient_b * lr
    if epoch % 100 == 0:
      print(f"第{epoch}轮，loss={loss:.5f}")
  return w,b,loss

train (x,y_actual,0.1,1000)
