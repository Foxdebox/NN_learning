# NN_learning
Rebuilding my understanding of neural networks from first principles.
这是一个记录我深度学习神经网络和机器学习过程的仓库。

# 目录
|算法/模型|状态|
|线性回归|学习中|


# 记录日志


## 线性回归
### v1:初次尝试根据输入值调整参数大小，使用了指数调整
-w和b不能分别调整，而且使用指数调整极易发散，不容易收敛
-[查看该版本代码](https://github.com/Foxdebox/NN_learning/commit/d6a8de7a271f5aa5fd4d6796cb7221723b2f2ca1)

### v2:引入了梯度下降和学习率
-参数=参数-梯度×学习率 加减取代了指数运算，训练更平滑，更易收敛
-w的梯度=loss × x_input
 b的梯度=loss × 1
 由此得以分别调整w和b
-learning rate 来调整步长，也可以让数值更稳定，更平滑更容易收敛
-手动输入x和y值有点麻烦和费时，下一步应该改进
-0.1的学习率太大了，在训练过程中容易参数爆炸，最好找到一种方法可以判断最佳学习率是多少
-[查看该版本代码](https://github.com/Foxdebox/NN_learning/commit/524b35c923bb204e3f5f121b35ede141365c50d3)
