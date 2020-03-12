#留一法进行交叉验证,返回最终的错误样本数
def cross_validation(feature, label):
    T = len(feature)
    count = 0
    for i in range(T):
        a = np.delete(feature, i, axis=0)
        b = np.delete(label, i)
        clf = tr.get_model(a, b)
        u = clf.predict([feature[i]])
        if (u != label[i]):
            count += 1
    return count

#绘制roc曲线
def ROC(a, b, m1, m2):
    x = [0.]
    y = [0.]
    t = 1
    for i in range(len(a)):
        if (a[i] == b[i]):
            x.append(x[t - 1])
            y.append(y[t - 1] + 1 / m1)
        else:
            x.append(x[t - 1] + 1 / m2)
            y.append(y[t - 1])
    plt.figure()
    plt.plot(x, y)
    plt.show()
