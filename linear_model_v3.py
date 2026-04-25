#让模型学习y=7x-3
data=[(1,4),(2,11),(3,18),(4,25),(5,32)] #可以换成别的数据
w = 1
b = 1
lr = 0.2 #可调（从大到小调整，如果参数爆炸就降低一个数量级）
def train(x,y_actual):
  global w,b,loss
  y_predict = w * x + b
  loss = y_predict - y_actual
  gradient_w = loss * x
  gradient_b = loss
  w = w - gradient_w * lr
  b = b - gradient_b * lr
  return w,b,loss
for epoch in range(100):
  for x,y_actual in data:
    train(x,y_actual)
  if epoch % 10 == 0:
   print(f"第{epoch}轮，w={w:.2f},b={b:.2f},误差={loss:.2f}")
# print() 的括号里可以用 f"普通文字{变量}普通文字"
# 这样可以直接显示变量的值，而不是把 {变量} 当作字符串输出
# :.2f表示保留两位小数
