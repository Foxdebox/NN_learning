w = 1
b = 1
lr = 0.1     #learing rate
def train(x,y_actual):
  global w,b
  y_predict = w * x + b
  loss = y_predict - y_actual
  gradient_w = loss * x
  gradient_b = loss
  w = w - gradient_w * lr
  b = b - gradient_b * lr
  print("y的预测值是",y_predict)
  print("误差是",loss)
  print("w调整至",w)
  print("b调整至",b)
  return w,b
while True:
  x = int(input("x的值是"))
  y_actual = int(input("y的真实值是"))
  train(x,y_actual)
