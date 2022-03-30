import numpy as np
import CleanData as clean
import pandas as pd
import matplotlib as plt
class TuiJian:
    def __init__(self,datas):
        self.X = datas
        self.M,self.N = self.X.shape
    #相似度计算
    @staticmethod
    def euclidean(inA, inB):#欧氏距离,inA,inB：默认列向量
        return 1.0/(1+np.linalg.norm(inA-inB))#取值范围[0,1]
    @staticmethod
    def pearson(inA, inB):#皮尔逊相关系数,inA,inB：默认列向量
        return ((1+np.corrcoef(inA, inB,rowvar=0))/2)[0,1]#取值范围[0,1]
    @staticmethod
    def cosine(inA,inB):#余弦相似度,inA,inB：默认列向量
        return ((1+(inA.T@inB)/(np.linalg.norm(inA)*np.linalg.norm(inB)))/2)#取值范围[0,1]
    #评分
    def assess(self,percentage=0.9,user=3,item=3):#在线评分
        U,S,V = np.linalg.svd(self.X)#奇异值分解数据。奇异值已排好序(降序),注意np.svd计算的V已转置
        # data = clean.CleanData(self.X, 0)  # 实例化
        # U,S,V = data.SVD() #这里的V未转置
        #选取前k个奇异值作为参考样本：按照前k个奇异值的平方和站总奇异值的平方和的百分来确定k值
        cumsum = np.cumsum(S)
        sum_list = cumsum/sum(S)
        k = np.where(sum_list>=percentage)[0][0]+1#选取占比90%的奇异值
        #构建k维奇异值分解:将原始数据转换维
        S = np.mat(np.diag(S[:k]))#k维对角矩阵
        #获取映射到k维空间的物品数据
        # r_data = (S.I@U[:,:k].T@self.X).T#V.T=(US).IX=S.I@U.T@X
        r_data = V.T[:k,:]# 列向量为特征向量
        #遍历已有评分的物品：预测item物品的评分--计算加权平均值，通过既有的评分加权求平均计算未知物品的评分
        weightMean,sum_similarity = [],0#相似度*评分，sum(相似度*评分)
        harmonic_mean = []
        for j in range(self.N):
            if self.X[user,j]==0 or j==item:#判断user是否对第j个物品评分
                continue
            #计算第j个物品与第item个物品的相似度
            similarity = self.pearson(r_data[:,j].T,r_data[:,item].T)#euclidean/pearson/cosine
            sum_similarity += similarity
            #计算加权平均值
            weightMean.append(similarity*self.X[user,j])#权值乘以评分
            harmonic_mean.append(similarity/self.X[user, j])  # 权值乘以评分
        pre_value = sum(weightMean)/sum_similarity#计算数学期望/加权平均值
        # pre_value = sum_similarity/sum(harmonic_mean)#加权调和平均值
        print('第'+str(item)+'个物品预测评分为：',np.round(pre_value,1))
        if pre_value==0:#如果相似度为0，则两着没有任何重合元素，终止本次循环
            return 0
        else:
            return pre_value
    def recommend(self,user,Num):#在线推荐
        #根据给定的用户，建立该用户未评分(评分==0)的物品栏
        # np.where(self.X[user,:]==0)
        no_score = np.nonzero(self.X[user,:]==0)[0]#返回
        if len(no_score)==0:#如果该用户对所有物品均已评分，则按评分降序排列，选择前Num个物品作推荐
            return np.argsort(-self.X[user,:])[:Num]
        #对该用分未评分的物品做预测评分
        score = []
        for j in range(len(no_score)):
            score.append(self.assess(user=user, item=j))
        #填充0评分物品
        self.X[user, no_score]=score
        #排序，为user推荐前Num个评分物品
        print(np.argsort(-self.X[user])[:Num])
        return np.argsort(-self.X[user])[:Num]

dataSet = [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
dataSet = [list(map(float, da)) for da in dataSet]# 浮点化数据
dataSet = np.array(dataSet)
def loadData(path,delim='\t'):
    with open(path,encoding='utf-8') as file:
        baseData = file.readlines()
    baseData = [list(map(float,da.strip().split(delim)))for da in baseData]
    return baseData
if __name__=='__main__':
    #加载训练集
    # path = '.\\Data\\ml-1m\\'
    # delim = '::'
    # user_data = pd.read_csv(path+'users.dat',sep=delim,names=['user_id','gender','item_id','rating','timestamp'],engine='python')
    # movies_data = pd.read_csv(path+'movies.dat',sep=delim,names=['item_id','item_name','type'],engine='python')
    # ratings_data = pd.read_csv(path+'ratings.dat',sep=delim,names=['user_id','item_id','rating','timestamp'],engine='python')
    #清洗数据

    #训练数据
    xx = TuiJian(dataSet)
    xx.recommend(user=0,Num=5)

    print()