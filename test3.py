import numpy as np
import pickle
import gzip
import os


# 初始化权重参数和偏置
weights = {} 
depth = 3  ##conv1的深度
fc_units=84  ##全连接层

## 网络结构
## Input(1*28*28)=>convlution(3*5*5)=>relu()=>maxpooling(3*3)=>flatten()=>fullconnected(84)=>Output(10)
weights["K1"] = 1e-2 * np.random.randn(1, depth, 5, 5).astype(np.float64)
weights["b1"] = np.zeros(depth).astype(np.float64)
weights["W2"] = 1e-2 * np.random.randn(depth * 11 * 11, fc_units).astype(np.float64)
weights["b2"] = np.zeros(fc_units).astype(np.float64)
weights["W3"] = 1e-2 * np.random.randn(fc_units, 10).astype(np.float64)
weights["b3"] = np.zeros(10).astype(np.float64)

# 初始化神经元和梯度
nuerons={}
gradients={}
#########################################################################

# 定义前向传播
def forward(X):
    nuerons["conv1"]=convolution_forward(X.astype(np.float64),weights["K1"],weights["b1"])
    nuerons["conv1_relu"]=relu_forward(nuerons["conv1"])
    
    nuerons["maxp1"]=maxpooling_forward(nuerons["conv1_relu"].astype(np.float64),pooling=(3,3))
    nuerons["flatten"]=flatten_forward(nuerons["maxp1"])
    
    nuerons["fc2"]=fullyconnected_forward(nuerons["flatten"],weights["W2"],weights["b2"])
    nuerons["fc2_relu"]=relu_forward(nuerons["fc2"])
    nuerons["y"]=fullyconnected_forward(nuerons["fc2_relu"],weights["W3"],weights["b3"])

    return nuerons["y"]

# 定义反向传播
def backward(X,y_true):
    loss,dy=cross_entropy_loss(nuerons["y"],y_true)
    gradients["W3"],gradients["b3"],gradients["fc2_relu"]=fullyconnected_backward(dy,weights["W3"],nuerons["fc2_relu"])
    gradients["fc2"]=relu_backward(gradients["fc2_relu"],nuerons["fc2"])
    
    gradients["W2"],gradients["b2"],gradients["flatten"]=fullyconnected_backward(gradients["fc2"],weights["W2"],nuerons["flatten"])
    gradients["maxp1"]=flatten_backward(gradients["flatten"],nuerons["maxp1"])
       
    gradients["conv1_relu"]=maxpooling_backward(gradients["maxp1"].astype(np.float64),nuerons["conv1_relu"].astype(np.float64),pooling=(3,3))
    gradients["conv1"]=relu_backward(gradients["conv1_relu"],nuerons["conv1"])
    gradients["K1"],gradients["b1"],_=convolution_backward(gradients["conv1"],weights["K1"],X)
    return loss
#########################################################################
# 获取精度
def get_accuracy(X,y_true):
    y_predict=forward(X)
    return np.mean(np.equal(np.argmax(y_predict,axis=-1), np.argmax(y_true,axis=-1)))
#########################################################################

#定义卷积层
def convolution_forward(X_input, Kernel, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param Kernel: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_X = np.lib.pad(X_input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    N, _, height, width = padding_X.shape
    C, D, k1, k2 = Kernel.shape

    ## 防止出现不能整除情况，用floor函数避免
    h_ = (height - k1) % strides[0]
    w_ = (width - k2) % strides[1]

    ##卷积之后的长度，padding为0
    H_ = 1 + (height - k1) // strides[0]
    W_ = 1 + (width - k2) // strides[1]
    conv_X = np.zeros((N, D, H_, W_))

    ##求和操作
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1):
                for w in np.arange(width - k2 + 1):
                    conv_X[n, d, h, w] = np.sum(padding_X[n, :, h:h + k1, w:w + k2] * Kernel[:, d]) + b[d]
    return conv_X

def convolution_backward(next_dX, Kernel, X, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积层的反向过程
    :param next_dX: 卷积输出层的梯度,(N,D,H',W'),H',W'为卷积输出层的高度和宽度
    :param Kernel: 当前层卷积核，(C,D,k1,k2)
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param padding: padding
    :param strides: 步长
    :return:
    """
    N, C, H, W = X.shape
    C, D, k1, k2 = Kernel.shape

    # 卷积核梯度
    padding_next_dX = Zeros_padding(next_dX, strides)
    # 增加高度和宽度0填充
    ppadding_next_dX = np.lib.pad(padding_next_dX, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant', constant_values=0)

    #旋转180度
    # 卷积核高度和宽度翻转180度
    flip_K = np.flip(Kernel, (2, 3))
    # 交换C,D为D,C；D变为输入通道数了，C变为输出通道数了
    switch_flip_K = np.swapaxes(flip_K, 0, 1)

    ##rot(180)*W
    dX = convolution_forward(ppadding_next_dX.astype(np.float64), switch_flip_K.astype(np.float64), np.zeros((C,), dtype=np.float64))

    # 求卷积核的梯度dK
    swap_W = np.swapaxes(X, 0, 1)  # 变为(C,N,H,W)与
    dW = convolution_forward(swap_W.astype(np.float64), padding_next_dX.astype(np.float64), np.zeros((D,), dtype=np.float64))

    # 偏置的梯度
    db = np.sum(np.sum(np.sum(next_dX, axis=-1), axis=-1), axis=0)  # 在高度、宽度上相加；批量大小上相加

    # 把padding减掉
    dX = Zeros_remove(dX, padding)

    return dW / N, db / N, dX
#########################################################################

"""
定义关于激活函数Relu的前向反向传播
"""
def relu_forward(X):
    """
    relu前向传播
    :param X: 待激活层
    :return: 激活后的结果
    """
    return np.maximum(0, X)


def relu_backward(next_dX, X):
    """
    relu反向传播
    :param next_dX: 激活后的梯度
    :param X: 激活前的值
    :return:
    """
    dX = np.where(np.greater(X, 0), next_dX, 0)
    return dX
#########################################################################


#向多维数组最后两位，每个行列之间增减指定的个数的零
#增加padding
def Zeros_padding(dX, strides):
    """
    :param dX: (N,D,H,W),H,W为卷积输出层的高度和宽度
    :param strides: 步长
    :return:
    """
    _, _, H, W = dX.shape
    pX = dX
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pX = np.insert(pX, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pX = np.insert(pX, w, 0, axis=3)
    return pX
#移除padding
def Zeros_remove(X, padding):
    """
    :param X: (N,C,H,W)
    :param paddings: (p1,p2)
    :return:
    """
    if padding[0] > 0 and padding[1] > 0:
        return X[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return X[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return X[:, :, :, padding[1]:-padding[1]]
    else:
        return X

#########################################################################
# 池化层，选择最大池化
def maxpooling_forward(X, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = X.shape
    # 零填充
    padding_X = np.lib.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # 输出的高度和宽度
    H_ = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    W_ = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_X = np.zeros((N, C, H_, W_))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(H_):
                for j in np.arange(W_):
                    pool_X[n, c, i, j] = np.max(padding_X[n, c, strides[0] * i:strides[0] * i + pooling[0], strides[1] * j:strides[1] * j + pooling[1]])
    return pool_X 

def maxpooling_backward(next_dX, X, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化反向过程
    :param next_dX：损失函数关于最大池化输出的损失
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = X.shape
    _, _, H_, W_ = next_dX.shape
    # 零填充
    padding_X = np.lib.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # 零填充后的梯度
    padding_dX = np.zeros_like(padding_X)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(H_):
                for j in np.arange(W_):
                    # 找到最大值的那个元素坐标，将梯度传给这个坐标
                    flat_idx = np.argmax(padding_X[n, c,strides[0] * i:strides[0] * i + pooling[0], strides[1] * j:strides[1] * j + pooling[1]])

                    h_idx = strides[0] * i + flat_idx // pooling[1]
                    w_idx = strides[1] * j + flat_idx % pooling[1]
                    padding_dX[n, c, h_idx, w_idx] += next_dX[n, c, i, j]
    # 返回时剔除零填充
    return Zeros_remove(padding_dX, padding)
#########################################################################

#将多维数组展平，前向传播
def flatten_forward(X):
    """
    :param X: 多维数组,形状(N,d1,d2,..)
    :return:
    """
    N = X.shape[0]
    return np.reshape(X, (N, -1))

#反向传播
def flatten_backward(next_dX, X):
    """
    :param next_dX:
    :param X:
    :return:
    """
    return np.reshape(next_dX, X.shape)
#########################################################################
#全连接层的前向传播
def fullyconnected_forward(X, W, b):
    """
    :param X: 当前层的输出,形状 (N,ln)
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :return: 下一层的输出
    """
    return np.dot(X, W) + b

#全连接层的反向传播
def fullyconnected_backward(next_dX, W, X):
    """
    :param next_dX: 下一层的梯度
    :param W: 当前层的权重
    :param X: 当前层的输出
    :return:
    """
    N = X.shape[0]
    delta = np.dot(next_dX, W.T)  # 当前层的梯度
    dw = np.dot(X.T, next_dX)  # 当前层权重的梯度
    db = np.sum(next_dX, axis=0)  # 当前层偏置的梯度, N个样本的梯度求和
    return dw / N, db / N, delta
#########################################################################


#########################################################################

#交叉熵损失函数
def cross_entropy_loss(y_predict, y_true):
    """
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值,shape(N,d)
    :return:
    """

    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1,keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    dy = y_probability - y_true
    return loss, dy
#########################################################################

##加载数据#########################################################################
def load_mnist_datasets(path='./Data/mnist.pkl.gz'):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' % path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')
        return train_set, val_set, test_set
def to_categorical(y, num_classes=None):
    """
    把类别标签转换为onehot编码
    源自keras
    Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
#加载数据，分类训练集，验证集，测试集
train_set, val_set, test_set = load_mnist_datasets('./Data/mnist.pkl.gz')
train_x,val_x,test_x=np.reshape(train_set[0],(-1,1,28,28)),np.reshape(val_set[0],(-1,1,28,28)),np.reshape(test_set[0],(-1,1,28,28))
train_y,val_y,test_y=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])
#########################################################################
# 随机选择训练样本
train_num = train_x.shape[0]
def train_select(batch_size):
    index=np.random.choice(train_num,batch_size)
    return train_x[index],train_y[index]

x,y= train_select(16)
print("x.shape:{},y.shape:{}".format(x.shape,y.shape))
#########################################################################

##随机梯度下降#############################################
def _copy_weights_to_zeros(weights):
        result = {}
        result.keys()
        for key in weights.keys():
            result[key] = np.zeros_like(weights[key])
        return result
class SGD(object):
    """
    小批量梯度下降法
    """
    ##初始化权重学习率动量因子迭代次数
    def __init__(self, weights, lr=0.01, momentum=0.9, decay=1e-5):
        """
        :param weights: 权重
        :param lr: 初始学习率
        :param momentum: 动量因子
        :param decay: 学习率衰减
        """
        self.v = _copy_weights_to_zeros(weights)  # 累积动量大小
        self.iterations = 0  # 迭代次数
        self.lr = self.init_lr = lr
        self.momentum = momentum
        self.decay = decay

    def iterate(self, weights, gradients):
        """
        迭代一次
        :param weights: 当前迭代权重
        :param gradients: 当前迭代梯度
        :return:
        """
        # 更新学习率
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        # 更新动量和梯度
        for key in self.v.keys():
            self.v[key] = self.momentum * self.v[key] + self.lr * gradients[key]
            weights[key] = weights[key] - self.v[key]

        # 更新迭代次数
        self.iterations += 1
##############################################################
##初始化变量
batch_size=4
steps = 2000  ##迭代
##初始化变量保存迭代步数step和loss值
steps_value = []
loss_value = []

# SGD更新梯度
sgd=SGD(weights,lr=0.01,decay=1e-6)

for s in range(steps):
    X,y=train_select(batch_size)
    
    forward(X)  # 前向过程
    loss=backward(X,y)  # 反向过程
    sgd.iterate(weights,gradients)  # 更新迭代次数

    if s % 100 ==0: # 每100次打印一次损失和正确率
        print("\n step:{:.2f}".format(s))
        print("\n loss:{}".format(loss))
        idx=np.random.choice(len(val_x),200)
        print(" train_acc:{}".format(get_accuracy(X,y)))
        print(" val_acc:{}".format(get_accuracy(val_x[idx],val_y[idx])))
        
        steps_value.append(s)
        loss_value.append(loss)
        
print("\n final result test_acc:{};  val_acc:{}".
      format(get_accuracy(test_x,test_y),get_accuracy(val_x,val_y)))

import matplotlib.pyplot as plt
#%matplotlib inline

##绘制loss值和迭代次数的曲线
plt.plot(steps_value, loss_value,color='green')
plt.xticks(rotation=45)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()
