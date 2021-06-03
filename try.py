from sklearn.datasets import load_iris
import numpy as np
import math
from collections import Counter


class decisionnode:
    def __init__(self, d=None, thre=None, results=None, NH=None, lb=None, rb=None, max_label=None):
        self.d = d   # d表示维度
        self.thre = thre  # thre表示二分时的比较值，将样本集分为2类
        self.results = results  # 最后的叶节点代表的类别
        self.NH = NH  # 存储各节点的样本量与经验熵的乘积，便于剪枝时使用
        self.lb = lb  # desision node,对应于样本在d维的数据小于thre时，树上相对于当前节点的子树上的节点
        self.rb = rb  # desision node,对应于样本在d维的数据大于thre时，树上相对于当前节点的子树上的节点
        self.max_label = max_label  # 记录当前节点包含的label中同类最多的label

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="round4", color='g')  # 定义判断结点形态
leafNode = dict(boxstyle="circle", color='#ADD8E6')  # 定义叶结点形态
arrow_args = dict(arrowstyle="<-", color='r')  # 定义箭头

# 绘制带箭头的注释


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 计算叶结点数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 计算树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center",
                        ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  # 在父子结点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制带箭头的注释
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,
                                       plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree, index=1):
    fig = plt.figure(index, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')



def Ent(y):

    if y.size > 1:
        category = list(set(y))
    else:
        category = [y.item()]
        y = [y.item()]

    ent = 0

    for label in category:
        p = len([label_ for label_ in y if label_ == label]) / len(y)
        ent += -p * math.log(p, 2)

    return ent



def Max_GainEnt(X, y, d):
    ent_X = Ent(y)
    X_attr = X[:, d]
    X_attr = list(set(X_attr))
    X_attr = sorted(X_attr)
    Gain = 0
    thre = 0

    for i in range(len(X_attr) - 1):
        thre_temp = (X_attr[i] + X_attr[i + 1]) / 2
        y_small_index = [i_arg for i_arg in range(
            len(X[:, d])) if X[i_arg, d] <= thre_temp]
        y_big_index = [i_arg for i_arg in range(
            len(X[:, d])) if X[i_arg, d] > thre_temp]
        y_small = y[y_small_index]
        y_big = y[y_big_index]

        Gain_temp = ent_X - (len(y_small) / len(y)) * \
            Ent(y_small) - (len(y_big) / len(y)) * Ent(y_big)
    
        if Gain < Gain_temp:
            Gain = Gain_temp
            thre = thre_temp
    return Gain, thre





def devide_group(X, y, thre, d):
    '''
    按照维度d下阈值为thre分为两类并返回
    '''
    X_in_d = X[:, d]
    x_small_index = [i_arg for i_arg in range(
        len(X[:, d])) if X[i_arg, d] <= thre]
    x_big_index = [i_arg for i_arg in range(
        len(X[:, d])) if X[i_arg, d] > thre]

    X_small = X[x_small_index]
    y_small = y[x_small_index]
    X_big = X[x_big_index]
    y_big = y[x_big_index]
    return X_small, y_small, X_big, y_big


def Mul_NH(y):
    ent = Ent(y)
    print('ent={},y_len={},all={}'.format(ent, len(y), ent * len(y)))
    return ent * len(y)


def maxlabel(y):
    label_ = Counter(y).most_common(1)
    return label_[0][0]

def attribute_based_on_GainEnt(X, y):
    D = np.arange(len(X[0]))
    Gain_max = 0
    thre_ = 0
    d_ = 0
    for d in D:
        Gain, thre = Max_GainEnt(X, y, d)
        if Gain_max < Gain:
            Gain_max = Gain
            thre_ = thre
            d_ = d  # 维度标号

    return Gain_max, thre_, d_


def buildtree(X, y):
    if y.size > 1:
         Gain_max, thre, d = attribute_based_on_GainEnt(X, y)
         if (Gain_max > 0 ):
            X_small, y_small, X_big, y_big = devide_group(X, y, thre, d)
            left_branch = buildtree(X_small, y_small)
            right_branch = buildtree(X_big, y_big)
            nh = Mul_NH(y)
            max_label = maxlabel(y)
            return decisionnode(d=d, thre=thre, NH=nh, lb=left_branch, rb=right_branch, max_label=max_label)
         else:
            nh = Mul_NH(y)
            max_label = maxlabel(y)
            return decisionnode(results=y[0], NH=nh, max_label=max_label)
    else:
        nh = Mul_NH(y)
        max_label = maxlabel(y)
        return decisionnode(results=y.item(), NH=nh, max_label=max_label)


def printtree(tree, indent='-', dict_tree={}, direct='L'):
    # 是否是叶节点

    if tree.results != None:
        print(tree.results)

        dict_tree = {direct: str(tree.results)}

    else:
        # 打印判断条件
        print(str(tree.d) + ":" + str(tree.thre) + "? ")
        # 打印分支
        print(indent + "L->",)

        a = printtree(tree.lb, indent=indent + "-", direct='L')
        aa = a.copy()
        print(indent + "R->",)

        b = printtree(tree.rb, indent=indent + "-", direct='R')
        bb = b.copy()
        aa.update(bb)
        stri = str(tree.d) + ":" + str(tree.thre) + "?"
        if indent != '-':
            dict_tree = {direct: {stri: aa}}
        else:
            dict_tree = {stri: aa}

    return dict_tree


def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.d]
        branch = None

        if v > tree.thre:
            branch = tree.rb
        else:
            branch = tree.lb

        return classify(observation, branch)


def pruning(tree, alpha=0.1):
    if tree.lb.results == None:
        pruning(tree.lb, alpha)
    if tree.rb.results == None:
        pruning(tree.rb, alpha)

    if tree.lb.results != None and tree.rb.results != None:
        before_pruning = tree.lb.NH + tree.rb.NH + 2 * alpha
        after_pruning = tree.NH + alpha
        print('before_pruning={},after_pruning={}'.format(
            before_pruning, after_pruning))
        if after_pruning <= before_pruning:
            print('pruning--{}:{}?'.format(tree.d, tree.thre))
            tree.lb, tree.rb = None, None
            tree.results = tree.max_label


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    permutation = np.random.permutation(X.shape[0])
    shuffled_dataset = X[permutation, :]
    shuffled_labels = y[permutation]

    train_data = shuffled_dataset[:100, :]
    train_label = shuffled_labels[:100]

    test_data = shuffled_dataset[100:150, :]
    test_label = shuffled_labels[100:150]

    tree2 = buildtree(train_data, train_label)

    b = printtree(tree=tree2)
    true_count = 0
    for i in range(len(test_label)):
        predict = classify(test_data[i], tree2)
        if predict == test_label[i]:
            true_count += 1
    print("C3Tree:{}".format(true_count))

    from pylab import *
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    import matplotlib.pyplot as plt
    createPlot(b, 1)
    # 剪枝处理

    pruning(tree=tree2, alpha=4)
    b = printtree(tree=tree2)
    true_count = 0
    for i in range(len(test_label)):
        predict = classify(test_data[i], tree2)
        if predict == test_label[i]:
            true_count += 1
    print("C3Tree:{}".format(true_count))
    createPlot(b, 2)
    plt.show()
