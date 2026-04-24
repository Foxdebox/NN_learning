w = 1
b = 1
def train(x_input,y_input):
 global w,b
 y_predict = w * x_input + b
 loss = y_input - y_predict
 w = w * 1.1 ** loss
 b = b * 1.1 ** loss
 print("预测值是",y_predict)
 print("相差",loss)
 print("w调整至", w)
 print("b调整至", b)
 return w,b
while True:
  x_input = int(input("输入x"))
  y_input = int(input("输入y"))
  train(x_input,y_input)
