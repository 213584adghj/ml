import numpy as np # 简单的线性回归 # 定义相关变量w, b = 0.,0.# 学习率lr = 0.0001points= np.genfromtxt("data.csv", delimiter=",")N = len(points)# 所有样本训练的次数epoch_number = 1000# 开始训练for epoch in range(epoch_number):    current_w, current_b = 0., 0.    # 计算所有样本的梯度平均值    for i in range(0, N):        current_w = current_w + 2/N * (w * points[i,0] - points[i,1])* points[i,0]        current_b = current_b + 2/N * (w * points[i,0] - points[i,1])    # 利用所有样本梯度的平均值更新w,b    w = w - lr * current_w    b = b - lr * current_b    # 每隔100步 计算一下当前的损失值    if epoch % 100 == 0:        current_loss = 0.        for i in range(0, N):            current_loss = current_loss + 1/N * (w * points[i, 0] + b - points[i, 1])**2        print('epoch :',epoch, 'current_loss:', current_loss)# 对所有样本迭代完100次后 输出最后的w,bprint('last weight w={0} and b={1}'.format(w,b))
w, b = 0.,0.
lr = 0.0001

points= np.genfromtxt("data.csv", delimiter=",")

N = len(points)
epoch_number = 1000
for epoch in range(epoch_number):

    current_w, current_b = 0., 0.
    for i in range(0, N):
        current_w = current_w + 2 / N * (w * points[i, 0] - points[i, 1]) * points[i, 0]
        current_b = current_b + 2 / N * (w * points[i, 0] - points[i, 1])
    w = w - lr * current_w

    b = b - lr * current_b
    if epoch % 100 == 0:        current_loss = 0.
    for i in range(0, N):
        current_loss = current_loss + 1 / N * (w * points[i, 0] + b - points[i, 1]) ** 2
    print('epoch :', epoch, 'current_loss:', current_loss)
print('last weight w={0} and b={1}'.format(w,b))
