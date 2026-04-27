#让模型学习y=7x-3
import numpy as np
def compute_mean_and_std(data):
  data = np.array(data)
  means = np.mean(data,axis=0)
  stds = np.std(data,axis=0)
  return means,stds
def normalize(x,means,stds):
  return (x-means)/stds
def train (x,y_actual):
  global w,b,loss
  y_predict = w * x  + b
  loss = y_predict - y_actual
  gradient_w= loss * x
  w = w- gradient_w * lr
  b = b - loss * lr
  return w,b,loss

data = [(1, 4), (2, 11), (3, 18), (4, 25), (5, 32), (6, 39), (7, 46), (8, 53), (9, 60), (10, 67)]
w = 1
b = 1
lr = 0.03
means,stds = compute_mean_and_std(data)


for epoch in range(100):
  for x,y_actual in data:
    nx = normalize(x,means[0],stds[0])
    ny_actual = normalize(y_actual,means[1],stds[1])
    train(nx,ny_actual)
  if epoch % 10 == 0:
    print(f"第{epoch}轮，w={w:.5f},b={b:.5f},loss={loss:.5f}")

w_original = w * (stds[1] / stds[0])
b_original = b * stds[1] + means[1] - w_original * means[0]

print(f"原始参数: w = {w_original:.5f}, b = {b_original:.5f}")
