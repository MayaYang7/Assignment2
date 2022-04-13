import numpy as np
import sklearn.datasets            
import matplotlib.pyplot as plt
import random

#The distance from data point to point
def PointToData(point,dataset):
    a = np.multiply(dataset - point,dataset - point)
    distance = np.sqrt(a[:,0]+a[:,1])
    return distance

#Select the initial k center clusters
def startpoint(k,dataset):
    m, n = np.shape(dataset)
    index1 = random.randint(0,len(dataset) - 1)
    A = []  # initial k center clusters
    A_dit = []  # initializ the distance from all points to the center cluster
    A.append(dataset[index1])
    sum_dis = np.zeros((m, 1))
    flag_mat = np.ones((m,1))
    flag_mat[index1] = 0
    for i in range(0, k - 1):
        A_dit.append((PointToData(A[i], dataset)).reshape(-1,1) )
        sum_dis =(sum_dis  + A_dit[i]) * flag_mat
        Index = np.argmax(sum_dis)
        flag_mat[Index] = 0
        A.append(dataset[Index])
    return A

#Download Data
Data = sklearn.datasets.load_iris()
dataset = Data.data[:,0:2]
dataset = np.array([[2, 4],
 [1.7, 2.8],
 [7, 8],
 [8.6, 8],
 [3.4, 1.5],
 [9,11]])


def classfy(dataset,Apoint):
    m,n = np.shape(dataset)
    dis_li = []
    num = 0
    for point in Apoint:
        distance = PointToData(point,dataset)
        dis_li.append(distance)
        if num == 0:
            dis_li_mat = dis_li[num]
        else:
            dis_li_mat = np.column_stack((dis_li_mat,dis_li[num]))
        num += 1
    result = np.argmin(dis_li_mat,axis=1)
    return result

#Seeking a new center for classification
def Center(dataset,label,k):
    i = 0
    newpoint = []
    for index in range(k):
        flag = (label==index)
        num = sum(flag)
        a = flag.reshape(-1,1) * dataset
        newpoint.append(np.sum(a,axis = 0)/num)
        i += 1
    return newpoint

#K-means主体函数
def myK(k,dataset):
    Startpoint = startpoint(k,dataset)
    m,n = np.shape(Startpoint)
    centerpoint = Startpoint
    labelset = classfy(dataset,Startpoint)
    newcenter = Center(dataset,labelset,k)
    print('out:cecnterpoint', centerpoint)
    print('out:newcenter', newcenter)
    flag = 0
    for i in range(k):
        for j in range(n):
            if centerpoint[i][j] != newcenter[i][j]:
                flag = 1
    while flag:
        print('inside:cecnterpoint', centerpoint)
        print('inside:newcenter', newcenter)
        flag = 0
        for i in range(k):
            for j in range(n):
                if centerpoint[i][j] != newcenter[i][j]:
                    flag = 1
        centerpoint = newcenter[:]
        labelset = classfy(dataset,centerpoint)
        newcenter = Center(dataset, labelset, k)
    return labelset,centerpoint

#testing 2 k-means
k=2
final_label,centerpoint = myK(k,dataset)
print('centerpoint:',centerpoint)
mat_center = np.mat(centerpoint)

#plot the graph
plt.scatter(dataset[:, 0], dataset[:, 1],40,10*(final_label+1))
plt.show()
