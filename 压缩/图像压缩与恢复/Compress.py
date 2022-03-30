import numpy as np
import CleanData as clean
import os
from PIL import Image
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def draw(img,p):
    plt.figure(1)
    plt.ion()
    plt.clf()
    plt.title('percent: {:.0%}'.format(p))
    plt.imshow(img)
    plt.pause(1)
    plt.ioff()  # 关闭交互模式
def extract(imgData,percentage=0.9):
    img = np.zeros_like(imgData)
    m,n,z = img.shape
    imgData = np.array(imgData)
    U,S,V = [],[],[]
    # 遍历通道，对各个通道做奇异值分解
    for i in range(z):
        u, s, v = np.linalg.svd(imgData[:,:,i])  # V已转置
        U.append(u)
        S.append(s)
        V.append(v)
    plt.figure(1)
    plt.ion()
    plt.clf()
    for p in np.linspace(0, 1, 11):#遍历奇异值百分比
        for i in range(z):#遍历通道
            # 选取前k个奇异值作为参考样本：按照前k个奇异值的平方和站总奇异值的平方和的百分来确定k值
            cumsum = np.cumsum(S[i])
            sum_list = cumsum / sum(S[i])
            k = np.where(sum_list >= p)[0][0]
            img[:,:,i]= U[i][:,:k]@np.diag(S[i][:k])@V[i][:k,:]
        #动态显示
        plt.title('percent: {:.0%}'.format(p))
        plt.imshow(img)
        plt.axis('off')
        plt.pause(1)
        plt.ioff()  # 关闭交互模式

def compress(path):
    #批量读取图片
    list_file = os.listdir(path)
    for fi in list_file:
        img = Image.open(path+'\\'+fi)
        extract(img)


if __name__=='__main__':
    path = r'.\Data'
    compress(path)