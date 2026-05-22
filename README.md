# NN_learning

Rebuilding my understanding of neural networks from first principles.

这是一个记录我深度学习神经网络和机器学习过程的仓库。

# 目录

| 算法/模型 | 状态 |
| :--- | :--- |
| 线性回归 | 学习中 |

---

# 记录日志

## 线性回归

### 🟢 v1: 初次尝试根据输入值调整参数大小，使用了指数调整
* w和b不能分别调整，而且使用指数调整极易发散，不容易收敛
* [🔗 查看该版本代码](https://github.com/Foxdebox/NN_learning/blob/d6a8de7a271f5aa5fd4d6796cb7221723b2f2ca1/linear_model_v1.py)

### 🟢 v2: 引入了梯度下降和学习率
* 参数=参数-梯度×学习率 加减取代了指数运算，训练更平滑，更易收敛
* w的梯度=loss × x_input, b的梯度=loss × 1, 由此得以分别调整w和b
* learning rate 来调整步长，也可以让数值更稳定，更平滑更容易收敛
* 手动输入x和y值有点麻烦和费时，下一步应该改进
* 0.1的学习率太大了，在训练过程中容易参数爆炸，最好找到一种方法可以判断最佳学习率是多少
* [🔗 查看该版本代码](https://github.com/Foxdebox/NN_learning/blob/524b35c923bb204e3f5f121b35ede141365c50d3/linear_model_v2.py)

### 🟢 v3: 自动化训练过程
* 移除了手动输入的模式，改为使用数据集进行循环训练
* 引入Epoch机制，可以进行任意次的迭代训练，更方便
* 使用f-string,并保留两位小数可以让输出更美观
* 相同学习率的情况下，使用数据的不同会影响是否参数爆炸。比如拟合y=7x-3时，如果data=[(1,4),(2,11),(3,18),(4,25),(5,32)]，甚至可以使用0.2的学习率来让他快速收敛；但是如果data=[(1,4),(2,11),(3,18),(4,25),(5,32),(6,39),(7,46),(8,53),(9,60),(10,67)]，0.1的学习率会就会使参数爆炸。
* 这是因为gradient = loss * x, 随着x变大，gradient也会变大，固定的学习率乘以大的梯度会导致挪动的距离太大，导致无法找到最优解。因此需要数据归一化。
* [🔗 查看该版本代码](https://github.com/Foxdebox/NN_learning/blob/8de22e3b5ace859fb6383ccc5b13a75597f7bf89/linear_model_v3.py)

### 🟢 v4: 增加了归一化和反归一化，以及调用NumPy库
* Normalization（归一化）：因为梯度和输入成正比，若不进行缩放，巨大的输入值会导致梯度异常增大，从而导致参数爆炸。通过将输入归一化到一致的较小的范围，可以避免这种干扰，让参数更新过程更稳定，高效。
* Denormalization(反归一化): 将模型计算出的参数还原回原本的单位和范围，让预测结果变得直观。
* NumPy:提供极其方便的内置函数(均值，标准差，极值等等)。此外，array(数组)让输入数据的处理变得非常方便
* [🔗 查看该版本代码](https://github.com/Foxdebox/NN_learning/blob/d69eb1dc619235b5ea50ddb501be3debc50008d9/linear_model_v4.py)

---

### 💡 阶段反思：使用AI帮助我发现并改正一些错误的认识
> * y_predict - y_actual 实际上是residual（残差)
> * loss 被叫做损失函数，是用来衡量预测值和真实值之间差距的函数，形式不止一种，包括但不限于
>   * MSE(mean square error) $L=1/N​\sum(y_{pred}​−y)^2$
>   * MAE(mean absolute error) $L=1/N\sum|y_{pred}-y|$ 等等等等
> * 梯度=loss对参数的导数（用链式法则求导）
> * 我之前的代码是SGD(stochastic gradient descent)随机梯度下降，是指每次用一个随机样本的梯度来更新参数。除此之外还有Batch GD(使用全部数据算梯度),Mini-batch(使用一小批算梯度)
> * 为什么平均梯度再更新更好？
> * 是因为目标是最小化整体的loss, 而L=所有样本的整体误差。如果使用SGD，每个样本都在拉着参数往自己方向走，非常容易来回震荡，不稳定。而使用Batch,则是看所有样本，求一个平均方向，再更新。既不容易来回震荡，也更接近真实目标（降低整体loss）
> * 这三种方法有各自优缺点
>   * SGD 快,有随机性，能跳出局部最优
>   * Batch 稳定但慢
>   * Mini-batch 折中
> * 向量化的向量并不是物理里有方向的量，而是一组数(ordered list)。比如X = [1, 2, 3, 4]就叫做向量。因此向量化的意思是用向量（数组）做计算，而不是单个的数。向量化的核心是将原本需要用循环执行的操作，转化为一次性对整个数组和矩阵的数学运算。

- [ ] **下一步目标**：向量化，使用平均梯度更新参数，理解并掌握需要使用的NumPy工具

---

### 🟢 v5: 将单变量变为多变量线性回归，实现向量化，NumPy的使用更熟练了
* **数学公式实现**：
  * MSE : $$Loss = \frac{1}{n} \sum_{i=1}^{n} (y_{pred} - y_{actual})^2$$
  * 对w的导数: $$\frac{\partial Loss}{\partial w} = \frac{2}{n} X^T (Xw + b - y)$$
  * 对b的导数: $$\frac{\partial Loss}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (y_{pred} - y_{actual})$$
  * $X^T$ 是对 x 的转置，即行变成列，列变成行。实现代码是 `x.T`。这么做是为了满足矩阵的乘法规则。
  * 归一化用的是 Z-Score, $$x' = \frac{x - \bar{x}}{\sigma}$$

* **NumPy 常用操作整理**：
  * `np.array(data[n])` 是获取第n行
  * `np.array(data[:,n])` 是获取第n列
  * `array[ , ]` 的规则：
    * "," 左边是行的范围，右边是列的范围
    * 只有 ":" 默认是全选。
    * ":" 左边是从哪开始，右边是到哪结束（左闭右开）。
    * ":" 左边没有数字代表从头开始取，右边没有数字代表取到最后一个。
    * `[:,:3]` 的意思是全选所有行，然后选中第0,1,2列。
    * `[:,-1]` 的意思是全选所有行，选中最后一列。
    * `[:,-2:]` 的意思是全选所有行，选中最后两列。
  * `(10,1)` 和 `(10,)` 是有区别的。前者是二维，后者是一维的。如果相减，会触发广播，最后的结果会是10*10的大矩阵。
  * `[:,-1]` 是 `(10,)`, `[:,-1:]` 是 `(10,1)`
  * `x.shape[]` 是获取行或者列的数量。比如 `x.shape[0]` 就是行数，`x.shape[1]` 就是列数
  * 矩阵乘法的代码是 `np.dot(A, B)`
  * `np.zeros(shape)` 是创建一个全零数组
  * `np.mean(data, axis=)` 是求平均值, `axis=0` 是压缩行，求每一列的平均值；`axis=1` 是压缩列，求每一行的平均值。如果不带 `axis=`，返回值就是一个标量。
  * `np.std(data, axis=)` 是求标准差，用法和 `np.mean()` 相同
* [🔗 查看该版本代码](https://github.com/Foxdebox/NN_learning/blob/f7cb1779ca842a0d9549d3ae2a56b1b69fb5adf3/linear_model_v5.py)

---

### 🟢 v6: 实战 Kaggle 房价预测 - 迈向真实世界数据

第一次处理真实世界的 CSV 数据（1460 行），并正式向 Kaggle 提交预测结果。在这个版本中，我实现了从 0.87 到 0.18 的分数突破。

#### 实战记录：
* **初次提交**：因为未对测试集的缺失值进行处理导致报错。
* **第二次提交**：因为未对 y 取对数，导致分数是 0.87504。
* **第三次提交**：将 y 进行对数转化之后，分数飞跃到 **0.18768**。

此次尝试只使用了三个特征值：`OverallQual`, `GrLivArea`, `TotalBsmtSF`。

因为预测误差应该看百分比而非绝对差值，为了避免高价房带偏模型，引入对数缩放：

###### 对数转换： 
$$y_{log} = \ln(1 + x)$$
（加 1 是为了防止 $x=0$）
* 代码实现：`np.log1p(y_actual)`

###### 指数逆转换： 
$$y_{pred} = e^x - 1$$
（将模型计算出的对数值还原回原本的单位和范围，让预测结果变得直观）
* 代码实现：`np.expm1(y_predict_log)`

#### Pandas常用操作
在处理 Kaggle 数据的过程中，我学习了 Pandas 的用法

##### **1. 数据读取与观察**
* `pd.read_csv('train.csv')`：将 CSV 文件转为 DataFrame 表格。
* `df.head(n)` / `df.tail(n)`：展示前 $n$ 行或后 $n$ 行。
* `df.shape`：告诉几行几列。
* `df.info()`：显示数据类型、非空数量、内存占用。
* `df.describe()`：计算每列的计数、均值、标准差、最小值、分位数、最大值。

##### **2. 缺失值处理**
* `df.isnull().sum()`：判断空值并求和，输出每一列有多少缺失值。
* `df.dropna()`：将有缺失值的行直接删掉。
* `df_test[features] = df_test[features].fillna(df[features].mean())`：用训练集的平均值来填补测试集的空值。

##### **3. 数据提取与转换**
* **目标值检查**：使用 `df['SalePrice'].describe()` 检查特定列。
* **钥匙逻辑**：`features = [...]` 和 `label = '...'` 就像制作钥匙，通过 `selected_df = df[features + label]` 提取特定列。
* **矩阵化**：`df.values` 将表格变成 NumPy array 形式，以便后续运算。

##### **4. 预测结果导出**
* `pd.DataFrame()`：创造一个空白表格。
* **新开列**：`submission_df['Id'] = df_test['Id']`。 将测试集的Id的值同步给它
* `to_csv('name.csv', index=False)`：将表格变成文件，`index=False` 用于取消行号。

**[🔗 查看该版本代码](https://github.com/Foxdebox/NN_learning/blob/c85c6ea36dd4f67f3d5215668287f05f44518ea6/linear_model_v6.py)**

---

## 非线性神经网络

### 🟢 Non-linear v1: 手动实现反向传播（学习中）

针对 2-3-1 结构神经网络的各个参数进行梯度计算与底层数学逻辑的推导记录。

#### 1. 网络架构与前向传播

该神经网络包含输入层、隐藏层、输出层共三层结构：

* **输入层**： 输入特征向量为

$$
X = (X_1, X_2)
$$

* **隐藏层**：
  * 左半边（线性组合）：

$$
Z_1 = XW_1 + b_1
$$

  * 右半边（非线性激活）：

$$
A_1 = ReLU(Z_1)
$$

  * 激活函数采用 **ReLU**：

$$
ReLU(x)=\max(0,x)=
\begin{cases}
x,&x>0\\
0,&x\le0
\end{cases}
$$

* **输出层**： 表达式为

$$
y = Z_2 = A_1W_2 + b_2
$$

* **损失函数（Loss）**： 采用平方损失函数形式

$$
Loss=(y-\hat y)^2
$$

#### 2. 手动求导与参数偏导数（链式法则）

利用链式法则，手推各个关键参数的偏导数公式：

* **W₂ 的偏导数**

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W_2} = 2(y-\hat y)\cdot A_1
$$

* **b₂ 的偏导数**

$$
\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial b_2} = 2(y-\hat y)
$$

* **W₁ 的偏导数**

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_1} = 2(y-\hat y)\cdot W_2\cdot ReLU'(x)\cdot X
$$

* **b₁ 的偏导数**

$$
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial b_1} = 2(y-\hat y)\cdot W_2\cdot ReLU'(x)
$$

* *注：ReLU 的导数满足：*

$$
ReLU'(x)=
\begin{cases}
1,&x>0\\
0,&x\le0
\end{cases}
$$

#### 3. 数学抽象与通用误差信号（Error Signal）

为了实现多层网络的通用化，引入 **误差信号（δ）** 的概念并提炼出核心通用公式：

* **黄金准则**：梯度的形状必须是原参数的形状。

* **参数梯度通用公式**（对于任意一层）

$$
Z=Input\times W+b
$$

  * W 的梯度：

$$
gradient_W=Input^T\times Error\_Signal
$$

  * b 的梯度：

$$
gradient_b=Error\_Signal
$$

* **误差信号定义与层间传递**

  * 输出层误差信号：

$$
\delta_n=\frac{\partial L}{\partial Z_n}
$$

  * 隐藏层误差信号：

$$
\delta_l=(\delta_{l+1}\times W_{l+1}^T)\odot\sigma'(Z_l)
$$

  * 其中：

    * $\delta_{l+1}$ 是下一层的误差信号
    * $W_{l+1}$ 是下一层的权重
    * $\odot$ 代表同形状矩阵对应位置一一相乘
    * $\sigma'(Z_l)$ 是当前层激活函数的导数

#### 4. NumPy 技巧与工程实现

* **ReLU 导数的代码实现**

```python
def relu_derivative(z):
    return (z > 0).astype(float)
```

* `return (z > 0)` 是判断是否大于零，得到布尔值
* `astype(float)` 是将布尔值转化为浮点数 0 或 1。 是变成 1，否变成 0
