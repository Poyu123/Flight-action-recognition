import os
import math
import time
import shutil
import random
import numpy as np
from PIL import Image
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from collectdata02 import RQDATA

#机头part==1    Brown	165 42 42	#A52A2A
#机翼part==2    CadetBlue1	152 245 255	#98F5FF
#尾翼part==3    LightGoldenrod4	139 129 76	#8B814C
#1 4 5 8 12 r
#2 3 6 7 10l
#1 2 3 4 9head
#5 6 7 8 11tail

def plot_scatter(points):
    # 提取x和y坐标
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    # 绘制散点图
    plt.scatter(x, y)
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('散点图')

    x=[int(i) for i in x]
    y=[int(i) for i in y]

    # 添加网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.xticks(range(min(x), max(x) + 2))
    plt.yticks(range(min(y), max(y) + 2))

    plt.show()

def insidetri(x, y, x1, y1, x2, y2, x3, y3):
    def cross_product(x1, y1, x2, y2, x3, y3):
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    area = cross_product(x1, y1, x2, y2, x3, y3)
    if area == 0:
        return False

    if area > 0:
        # 点按逆时针排列
        return (cross_product(x1, y1, x2, y2, x, y) >= 0 and
                cross_product(x2, y2, x3, y3, x, y) >= 0 and
                cross_product(x3, y3, x1, y1, x, y) >= 0)
    else:
        # 点按顺时针排列
        return (cross_product(x1, y1, x2, y2, x, y) <= 0 and
                cross_product(x2, y2, x3, y3, x, y) <= 0 and
                cross_product(x3, y3, x1, y1, x, y) <= 0)

def getborder (x1, y1, x2, y2, x3, y3,sizepic_x,sizepic_y):
    max_x=x1
    max_y=y1
    min_x=x1
    min_y=y1

    if x2>max_x:
        max_x=x2
    elif x2<min_x:
        min_x=x2
    
    if x3>max_x:
        max_x=x2
    elif x3<min_x:
        min_x=x2
    
    if y2>max_y:
        max_y=y2
    elif y2<min_y:
        min_y=y2
    
    if y3>max_y:
        max_y=y3
    elif y3<min_y:
        min_y=y3
    
    max_x=math.ceil(max_x)+1
    min_x=math.ceil(min_x)-1
    max_y=math.ceil(max_y)+1
    min_y=math.ceil(min_y)-1

    if min_x<0:
        min_x=0
    if min_y<0:
        min_y=0
    
    if max_x>=sizepic_x:
        max_x=sizepic_x-1
    if max_y>=sizepic_y:
        max_y=sizepic_y-1

    #返回的是index
    return max_x,min_x,max_y,min_y
#生成图片
def produceimage (r,g,b,output_dir):

    r = r.astype(np.uint8)
    g = g.astype(np.uint8)
    b = b.astype(np.uint8)

    rgb_image = np.dstack((r, g, b))
    image = Image.fromarray(rgb_image)
    image.save(output_dir)

def mean_data(X):
    # 计算均值并中心化数据
    means = np.mean(X, axis=0)
    X_centered = X - means
    return X_centered, means

def covariance_matrix(X_centered):
    # 计算协方差矩阵
    cov = np.cov(X_centered, rowvar=False)
    return cov

def pca(X_centered, n_components):
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(np.cov(X_centered, rowvar=False))

    # 将特征值和特征向量按特征值大小降序排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前n_components个特征向量
    eigenvectors = eigenvectors[:, :n_components]

    # 投影数据到主成分空间
    X_projected = X_centered.dot(eigenvectors)

    return X_projected, eigenvalues, eigenvectors

def plane_model (size_plane,model):

    if model == 1 :
        vectors = np.row_stack([
            np.array([0.0 + size_plane,0.0 + size_plane, 0.0 - size_plane]),#1
            np.array([0.0 + size_plane,0.0 - size_plane, 0.0 - size_plane]),#2
            np.array([0.0 + size_plane,0.0 - size_plane, 0.0 + size_plane]),#3
            np.array([0.0 + size_plane,0.0 + size_plane, 0.0 + size_plane]),#4
            np.array([0.0 - size_plane,0.0 + size_plane, 0.0 - size_plane]),#5
            np.array([0.0 - size_plane,0.0 - size_plane, 0.0 - size_plane]),#6
            np.array([0.0 - size_plane,0.0 - size_plane, 0.0 + size_plane]),#7
            np.array([0.0 - size_plane,0.0 + size_plane, 0.0 + size_plane]),#8
            np.array([0.0 + (size_plane*2),0.0, 0.0 + size_plane]),#9
            np.array([0.0 - size_plane,0.0 - (size_plane*2), 0.0 - size_plane]),#10
            np.array([0.0 - size_plane,0.0, 0.0 - (size_plane*2)]),#11
            np.array([0.0 - size_plane,0.0 + (size_plane*2), 0.0 - size_plane])#12
        ])
    elif model == 2 :
        vectors = np.row_stack([
            np.array([0.0 + size_plane,0.0 + size_plane, 0.0 - size_plane]),#1
            np.array([0.0 + size_plane,0.0 - size_plane, 0.0 - size_plane]),#2
            np.array([0.0 + size_plane,0.0 - size_plane, 0.0 + size_plane]),#3
            np.array([0.0 + size_plane,0.0 + size_plane, 0.0 + size_plane]),#4
            np.array([0.0 - size_plane,0.0 + size_plane, 0.0 - size_plane]),#5
            np.array([0.0 - size_plane,0.0 - size_plane, 0.0 - size_plane]),#6
            np.array([0.0 - size_plane,0.0 - size_plane, 0.0 + size_plane]),#7
            np.array([0.0 - size_plane,0.0 + size_plane, 0.0 + size_plane]),#8
            np.array([0.0 + (size_plane*3),0.0, 0.0 + size_plane]),#9
            np.array([0.0 - 0.6*size_plane,0.0 - (size_plane*5), 0.0 - 0.75*size_plane]),#10
            np.array([0.0 - 1.5*size_plane,0.0, 0.0 - (size_plane*2)]),#11
            np.array([0.0 - 0.6*size_plane,0.0 + (size_plane*5), 0.0 - 0.75*size_plane])#12
        ])
    elif model==0 :
        vectors = np.row_stack([
            np.array([0.0,0.0,0.0]),#1
            np.array([0.0,0.0,0.0]),#2
            np.array([0.0,0.0,0.0]),#3
            np.array([0.0,0.0,0.0]),#4
            np.array([0.0,0.0,0.0]),#5
            np.array([0.0,0.0,0.0]),#6
            np.array([0.0,0.0,0.0]),#7
            np.array([0.0,0.0,0.0]),#8
            np.array([0.0,0.0,0.0]),#9
            np.array([0.0,0.0,0.0]),#10
            np.array([0.0,0.0,0.0]),#11
            np.array([0.0,0.0,0.0])#12
        ])


    return vectors

def process_flight_data(source_dir,ratio):

    size_plane_1=0
    size_plane_2=0
    size_plane_3=0

    model=2#飞机模型的样式
    ratio=ratio#飞机尺寸大小

    # 读取数据
    longitude, latitude, height, roll, pitch, yaw, time= RQDATA(source_dir)
    
    # 转换经纬度到笛卡尔坐标系
    
    earth_radius = 6371000  # 地球的平均半径，单位：米
    x = []
    y = []
    z = []
    for i in range(len(time)):

        phi = np.radians(latitude[i])
        lambda_ = np.radians(longitude[i])

        x.append((earth_radius + height[i]) * np.cos(phi) * np.cos(lambda_))
        y.append((earth_radius + height[i]) * np.cos(phi) * np.sin(lambda_))
        z.append((earth_radius + height[i]) * np.sin(phi))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    '''
    x = np.array(longitude)
    y = np.array(latitude)
    z = np.array(height)
    '''

    X = np.column_stack((x, y, z))

    # 去中心化数据，便于后续投影
    X_centered, means = mean_data(X)
    x = X_centered[:, 0]  # 获取第一列（第一个特征的所有样本）
    y = X_centered[:, 1]  # 获取第二列（第二个特征的所有样本）
    z = X_centered[:, 2]  # 获取第三列（第三个特征的所有样本）
    
    # 计算协方差矩阵
    cov = covariance_matrix(X_centered)

    # PCA分析
    n_components = 2  # 保留2个主成分
    X_projected, eigenvalues, eigenvectors = pca(X_centered, n_components)

    # 归一化特征向量
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    
    eigenvectors1=eigenvectors[:,0]
    eigenvectors2=eigenvectors[:,1]
    eigenvectors3_pre=np.cross(eigenvectors1, eigenvectors2)
    norm = np.linalg.norm(eigenvectors3_pre)
    eigenvectors3 = eigenvectors3_pre / norm

    eigenvectors_xz=np.column_stack((eigenvectors1, eigenvectors3))
    eigenvectors_yz=np.column_stack(( eigenvectors2, eigenvectors3))

    ARR1=[]
    ARR2=[]
    ARR3=[]

    # 投影和旋转
    for i in range(0, len(time)):
        roll_angle = np.radians(roll[i])
        pitch_angle = np.radians(pitch[i])
        yaw_angle = np.radians(yaw[i])

        Rz_yaw = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle), 0],
                           [np.sin(yaw_angle), np.cos(yaw_angle), 0],
                           [0, 0, 1]])
        Ry_pitch = np.array([[np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                             [0, 1, 0],
                             [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]])
        Rx_roll = np.array([[1, 0, 0],
                            [0, np.cos(roll_angle), -np.sin(roll_angle)],
                            [0, np.sin(roll_angle), np.cos(roll_angle)]])
        R = Rz_yaw @ Ry_pitch @ Rx_roll

        vectors=plane_model (size_plane_1,0)
        vector_rotated = vectors @ R.T
        vector_rotated[:,0]+= x[i]
        vector_rotated[:,1]+= y[i]
        vector_rotated[:,2]+= z[i]
        xy = vector_rotated @ eigenvectors

        del vector_rotated
        del vectors

        vectors=plane_model (size_plane_2,0)
        vector_rotated = vectors @ R.T
        vector_rotated[:,0]+= x[i]
        vector_rotated[:,1]+= y[i]
        vector_rotated[:,2]+= z[i]
        xz=vector_rotated@eigenvectors_xz

        del vector_rotated
        del vectors

        vectors=plane_model (size_plane_3,0)
        vector_rotated = vectors @ R.T
        vector_rotated[:,0]+= x[i]
        vector_rotated[:,1]+= y[i]
        vector_rotated[:,2]+= z[i]
        yz=vector_rotated@eigenvectors_yz

        ARR1.append(xy)
        ARR2.append(xz)
        ARR3.append(yz)
    
    a1,b1=findmax_xandy(ARR1)
    a2,b2=findmin_xandy(ARR1)

    size_plane_1=ratio*round(max((a1-a2),(b1-b2)))

    a1,b1=findmax_xandy(ARR2)
    a2,b2=findmin_xandy(ARR2)

    size_plane_2=ratio*round(max((a1-a2),(b1-b2)))

    a1,b1=findmax_xandy(ARR3)
    a2,b2=findmin_xandy(ARR3)

    size_plane_3=ratio*round(max((a1-a2),(b1-b2)))
    del ARR1
    del ARR2
    del ARR3

    ARR1=[]
    ARR2=[]
    ARR3=[]
    # 投影和旋转
    for i in range(0, len(time)):
        roll_angle = np.radians(roll[i])
        pitch_angle = np.radians(pitch[i])
        yaw_angle = np.radians(yaw[i])

        Rz_yaw = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle), 0],
                           [np.sin(yaw_angle), np.cos(yaw_angle), 0],
                           [0, 0, 1]])
        Ry_pitch = np.array([[np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                             [0, 1, 0],
                             [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]])
        Rx_roll = np.array([[1, 0, 0],
                            [0, np.cos(roll_angle), -np.sin(roll_angle)],
                            [0, np.sin(roll_angle), np.cos(roll_angle)]])
        R = Rz_yaw @ Ry_pitch @ Rx_roll

        vectors=plane_model (size_plane_1,model)
        vector_rotated = vectors @ R.T
        vector_rotated[:,0]+= x[i]
        vector_rotated[:,1]+= y[i]
        vector_rotated[:,2]+= z[i]
        xy = vector_rotated @ eigenvectors

        del vector_rotated
        del vectors

        vectors=plane_model (size_plane_2,model)
        vector_rotated = vectors @ R.T
        vector_rotated[:,0]+= x[i]
        vector_rotated[:,1]+= y[i]
        vector_rotated[:,2]+= z[i]
        xz=vector_rotated@eigenvectors_xz

        del vector_rotated
        del vectors

        vectors=plane_model (size_plane_3,model)
        vector_rotated = vectors @ R.T
        vector_rotated[:,0]+= x[i]
        vector_rotated[:,1]+= y[i]
        vector_rotated[:,2]+= z[i]
        yz=vector_rotated@eigenvectors_yz

        ARR1.append(xy)
        ARR2.append(xz)
        ARR3.append(yz)
    '''
    print(ARR1)
    print()
    print(ARR2)
    print()
    print(ARR3)
    print()
    '''
    
    return ARR1,ARR2,ARR3

def diss(x1,y1,x2,y2):
    ddis=(x1-x2)**2+(y1-y2)**2
    return ddis

def Left_Triangle_find(mx):
    x1=[]
    y1=[]
    x1.append(mx[0][0])
    x1.append(mx[3][0])
    x1.append(mx[4][0])
    x1.append(mx[7][0])
    x1.append(mx[11][0])
    
    y1.append(mx[0][1])
    y1.append(mx[3][1])
    y1.append(mx[4][1])
    y1.append(mx[7][1])
    y1.append(mx[11][1])
    Dis=[]
    Min=114514111
    mid=0
    for i in range(0,5):
        Dis.append(diss(x1[i],y1[i],x1[0],y1[0]))
        for j in range(1,5):
            Dis[i]=Dis[i]+diss(x1[i],y1[i],x1[j],y1[j])
        if Dis[i]<Min:
            Min=Dis[i]
            mid=i
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]
    tag=0
    for i in range(0,5):
        # if i==mid:
            # continue
        for j in range(0,5):
            # if i!=j and j!=mid:
            if j!=i:
                for k in range(0,5):
                    if k!=j and k!=i:
                        X1.append(x1[i])
                        Y1.append(y1[i])
                        X2.append(x1[j])
                        Y2.append(y1[j])
                        X3.append(x1[k])
                        Y3.append(y1[k])
    return X1,Y1,X2,Y2,X3,Y3,2

def Right_Triangle_find(mx):
    x1=[]
    y1=[]
    x1.append(mx[1][0])
    x1.append(mx[2][0])
    x1.append(mx[5][0])
    x1.append(mx[6][0])
    x1.append(mx[9][0])
    
    y1.append(mx[1][1])
    y1.append(mx[2][1])
    y1.append(mx[5][1])
    y1.append(mx[6][1])
    y1.append(mx[9][1])
    Dis=[]
    Min=114514111
    mid=0
    for i in range(0,5):
        Dis.append(diss(x1[i],y1[i],x1[0],y1[0]))
        for j in range(1,5):
            Dis[i]=Dis[i]+diss(x1[i],y1[i],x1[j],y1[j])
        if Dis[i]<Min:
            Min=Dis[i]
            mid=i
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]
    tag=0
    for i in range(0,5):
        # if i==mid:
            # continue
        for j in range(0,5):
            # if i!=j and j!=mid:
            if j!=i:
                for k in range(0,5):
                    if k!=j and k!=i:
                        X1.append(x1[i])
                        Y1.append(y1[i])
                        X2.append(x1[j])
                        Y2.append(y1[j])
                        X3.append(x1[k])
                        Y3.append(y1[k])
    return X1,Y1,X2,Y2,X3,Y3,2

def Head_find(mx):
    #1 2 3 4 9head
    x1=[]
    y1=[]
    x1.append(mx[0][0])
    x1.append(mx[1][0])
    x1.append(mx[2][0])
    x1.append(mx[3][0])
    x1.append(mx[8][0])
    
    y1.append(mx[0][1])
    y1.append(mx[1][1])
    y1.append(mx[2][1])
    y1.append(mx[3][1])
    y1.append(mx[8][1])
    #1 6 5 8 10
    # x1.append(mx[1][0])
    # x1.append(mx[2][0])
    # x1.append(mx[5][0])
    # x1.append(mx[6][0])
    # x1.append(mx[9][0])
    
    # y1.append(mx[1][1])
    # y1.append(mx[2][1])
    # y1.append(mx[5][1])
    # y1.append(mx[6][1])
    # y1.append(mx[9][1])
    Dis=[]
    Min=1145141
    mid=0
    for i in range(0,5):
        Dis.append(diss(x1[i],y1[i],x1[0],y1[0]))
        for j in range(1,5):
            Dis[i]=Dis[i]+diss(x1[i],y1[i],x1[j],y1[j])
        if Dis[i]<Min:
            Min=Dis[i]
            mid=i
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]
    tag=0
    for i in range(0,5):
        # if i==mid:
            # continue
        for j in range(0,5):
            # if i!=j and j!=mid:
            if j!=i:
                for k in range(0,5):
                    if k!=j and k!=i:
                        X1.append(x1[i])
                        Y1.append(y1[i])
                        X2.append(x1[j])
                        Y2.append(y1[j])
                        X3.append(x1[k])
                        Y3.append(y1[k])
    return X1,Y1,X2,Y2,X3,Y3,1

def Tail_Triangle_find(mx):
    x1=[]
    y1=[]
    #1 2 5 6 11
    x1.append(mx[0][0])
    x1.append(mx[1][0])
    x1.append(mx[4][0])
    x1.append(mx[5][0])
    x1.append(mx[10][0])
    
    y1.append(mx[0][1])
    y1.append(mx[1][1])
    y1.append(mx[4][1])
    y1.append(mx[5][1])
    y1.append(mx[10][1])
    # x1.append(mx[1][0])
    # x1.append(mx[2][0])
    # x1.append(mx[5][0])
    # x1.append(mx[6][0])
    # x1.append(mx[9][0])
    
    # y1.append(mx[1][1])
    # y1.append(mx[2][1])
    # y1.append(mx[5][1])
    # y1.append(mx[6][1])
    # y1.append(mx[9][1])
    Dis=[]
    Min=114514111
    mid=0
    for i in range(0,5):
        Dis.append(diss(x1[i],y1[i],x1[0],y1[0]))
        for j in range(1,5):
            Dis[i]=Dis[i]+diss(x1[i],y1[i],x1[j],y1[j])
        if Dis[i]<Min:
            Min=Dis[i]
            mid=i
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]
    tag=0
    for i in range(0,5):
        # if i==mid:
            # continue
        for j in range(0,5):
            # if i!=j and j!=mid:
            if j!=i:
                for k in range(0,5):
                    if k!=j and k!=i:
                        X1.append(x1[i])
                        Y1.append(y1[i])
                        X2.append(x1[j])
                        Y2.append(y1[j])
                        X3.append(x1[k])
                        Y3.append(y1[k])
    return X1,Y1,X2,Y2,X3,Y3,3

def findmin_xandy(ARR):
    minx=ARR[0][0][0]
    miny=ARR[0][0][1]

    for mx in ARR :
        counter=0
        while(counter<12):
            if mx[counter][0]<minx :
                minx=mx[counter][0]
            if mx[counter][1]<miny :
                miny=mx[counter][1]
            counter+=1
    return minx, miny

def findmax_xandy(ARR):
    maxx=ARR[0][0][0]
    maxy=ARR[0][0][1]

    for mx in ARR :
        counter=0
        while(counter<12):
            if mx[counter][0]>maxx :
                maxx=mx[counter][0]
            if mx[counter][1]>maxy :
                maxy=mx[counter][1]
            counter+=1
    return maxx, maxy

def getnewARR(ARR,num_p,sizepic_x,sizepic_y):

    minx,miny=findmin_xandy(ARR)
    for i in range(len(ARR)):
        for m in range(12):
            ARR[i][m][0]-=minx
            ARR[i][m][1]-=miny
    maxx,maxy=findmax_xandy(ARR)
    for i in range(len(ARR)):
        for m in range(12):
            ARR[i][m][0]=ARR[i][m][0] / maxx * sizepic_x
            ARR[i][m][1]=ARR[i][m][1] / maxy * sizepic_y
    ARR_new=[]
    #print(len(ARR))
    step=len(ARR)//num_p
    #print('step is '+str(step))
    counter=0
    while(counter<(num_p-1)):
        #print("index is "+str(counter*step))
        ARR_new.append(ARR[counter*step])
        counter+=1
    ARR_new.append(ARR[-1])

    return ARR_new

def endback02(source_dir,NUM_L):
    name01=os.listdir(os.path.join(source_dir,'temptLis'+str(NUM_L)))
    for name01_t in name01:
        name02=os.listdir(os.path.join(source_dir,'temptLis'+str(NUM_L),name01_t))
        for name02_t in name02 :
            shutil.move(os.path.join(source_dir,'temptLis'+str(NUM_L),name01_t,name02_t),os.path.join(source_dir,name01_t,name02_t))
    for name01_t in name01:
        os.rmdir(os.path.join(source_dir,'temptLis'+str(NUM_L),name01_t))
    os.rmdir(os.path.join(source_dir,'temptLis'+str(NUM_L)))   

def presolve01(source_dir,NUM_L,mode=1):

    name01=os.listdir(source_dir)
    i=0
    while (name01[i] == ('temptLis'+str(NUM_L))):
        i+=1
    min=len(os.listdir(os.path.join(source_dir,name01[i])))

    if mode == 2 :
        for name01_t in name01:
            if name01_t != ('temptLis'+str(NUM_L)) and min > len(os.listdir(os.path.join(source_dir,name01_t))):
                min=len(os.listdir(os.path.join(source_dir,name01_t)))
        return min

    if mode:
        os.mkdir(os.path.join(source_dir,'temptLis'+str(NUM_L)))

    total_num=0
    for name01_t in name01:
        if name01_t != ('temptLis'+str(NUM_L)):
            name02=os.listdir(os.path.join(source_dir,name01_t))
            total_num+=len(name02)
    progress_bar=tqdm(total=total_num)

    for name in name01:
        if name != ('temptLis'+str(NUM_L)) and mode:
            os.mkdir(os.path.join(source_dir,'temptLis'+str(NUM_L),name))
    for name01_t in name01:
        if name01_t != ('temptLis'+str(NUM_L)):
            name02=os.listdir(os.path.join(source_dir,name01_t))
            for name02_t in name02 :
                file_path=os.path.join(source_dir,name01_t,name02_t)
                with open(file_path, 'r', encoding='utf-8') as file:
                    length=len(file.readlines())-1
                if length<NUM_L:
                    shutil.move(file_path,os.path.join(source_dir,'temptLis'+str(NUM_L),name01_t,name02_t))
                progress_bar.update(1)

    progress_bar.close()

    for name01_t in name01:
        if name01_t != ('temptLis'+str(NUM_L)) and min > len(os.listdir(os.path.join(source_dir,name01_t))):
            min=len(os.listdir(os.path.join(source_dir,name01_t)))
    
    return min

def PREMAIN (NUM_C,NUM_L,ifback,source_dir):

    start_time=time.time()
    print('进行中...')

    NUM_A=len(os.listdir(source_dir))#分类数量

    NUM_L_T=0

    for name01_t in os.listdir(source_dir):
        if 'tempt' in name01_t:
            NUM_L_T=int(name01_t[8:]) 
            break

    ps_mode=1

    if NUM_L_T:
        NUM_A-=1
        if (NUM_L_T > NUM_L):
            endback02(source_dir,NUM_L_T)
        elif NUM_L_T < NUM_L :
            os.rename(os.path.join(source_dir,'temptLis'+str(NUM_L_T)), os.path.join(source_dir,'temptLis'+str(NUM_L)))
            ps_mode=0
        elif NUM_L_T == NUM_L:
            ps_mode=2

    nummin=presolve01(source_dir,NUM_L,mode=ps_mode)

    print('允许设置 NUM_C 的最大值：'+str(nummin))
    if NUM_C>nummin or NUM_C==0:
        NUM_C=nummin

    if ifback:
        endback02(source_dir,NUM_L)

    print('已完成数据初步筛选。')

    return NUM_A

def numdigits(num):
    counter=0
    while(num>=1):
        num/=10
        counter+=1
    return counter

def MAIN(source_dir,output_dir,num_p,ratio_p,color_arr,sizepic_x,sizepic_y,mode_level):

    ARR1,ARR2,ARR3=process_flight_data(source_dir,ratio_p)

    ARR1_new=getnewARR(ARR1,num_p,sizepic_x,sizepic_y)
    ARR2_new=getnewARR(ARR2,num_p,sizepic_x,sizepic_y)
    ARR3_new=getnewARR(ARR3,num_p,sizepic_x,sizepic_y)

    r = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    g = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    b = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)

    X1ALL,Y1ALL,X2ALL,Y2ALL,X3ALL,Y3ALL,MODE=[],[],[],[],[],[],[]
    mode=0

    for mx in ARR3_new:
        if mode_level<3:
            break

        X1,Y1,X2,Y2,X3,Y3,mode=Left_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Right_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Tail_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Head_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3


    for i in range(len(X1ALL)):

        x1, y1, x2, y2, x3, y3=X1ALL[i],Y1ALL[i],X2ALL[i],Y2ALL[i],X3ALL[i],Y3ALL[i]
        max_x,min_x,max_y,min_y=getborder(x1, y1, x2, y2, x3, y3,sizepic_x,sizepic_y)
        color=color_arr[0]
        n=min_y
        m=min_x
        while(n<max_y):
            m=min_x
            while(m<max_x):
                if insidetri(m ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n+1 , x1, y1, x2, y2, x3, y3) or insidetri(m ,n+1 , x1, y1, x2, y2, x3, y3) :
                    if MODE[i] ==1 :
                        r[m][n]=color
                    elif MODE[i] ==2 :
                        g[m][n]=color
                    elif MODE[i] ==3 :
                        b[m][n]=color
                m+=1
            n+=1

    X1ALL,Y1ALL,X2ALL,Y2ALL,X3ALL,Y3ALL,MODE=[],[],[],[],[],[],[]
    mode=0

    for mx in ARR2_new:
        if mode_level<2:
            break

        X1,Y1,X2,Y2,X3,Y3,mode=Left_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Right_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Tail_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Head_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

    for i in range(len(X1ALL)):

        x1, y1, x2, y2, x3, y3=X1ALL[i],Y1ALL[i],X2ALL[i],Y2ALL[i],X3ALL[i],Y3ALL[i]
        color=color_arr[1]
        max_x,min_x,max_y,min_y=getborder(x1, y1, x2, y2, x3, y3,sizepic_x,sizepic_y)
        n=min_y
        m=min_x
        while(n<max_y):
            m=min_x
            while(m<max_x):
                if insidetri(m ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n+1 , x1, y1, x2, y2, x3, y3) or insidetri(m ,n+1 , x1, y1, x2, y2, x3, y3) :
                    if MODE[i] ==1 :
                        r[m][n]=color
                    elif MODE[i] ==2 :
                        g[m][n]=color
                    elif MODE[i] ==3 :
                        b[m][n]=color
                m+=1
            n+=1

    X1ALL,Y1ALL,X2ALL,Y2ALL,X3ALL,Y3ALL,MODE=[],[],[],[],[],[],[]
    mode=0

    for mx in ARR1_new:

        X1,Y1,X2,Y2,X3,Y3,mode=Left_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Right_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Tail_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Head_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

    for i in range(len(X1ALL)):

        x1, y1, x2, y2, x3, y3=X1ALL[i],Y1ALL[i],X2ALL[i],Y2ALL[i],X3ALL[i],Y3ALL[i]
        color=color_arr[2]
        max_x,min_x,max_y,min_y=getborder(x1, y1, x2, y2, x3, y3,sizepic_x,sizepic_y)
        n=min_y
        m=min_x
        while(n<max_y):
            m=min_x
            while(m<max_x):
                if insidetri(m ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n+1 , x1, y1, x2, y2, x3, y3) or insidetri(m ,n+1 , x1, y1, x2, y2, x3, y3) :
                    if MODE[i] ==1 :
                        r[m][n]=color
                    elif MODE[i] ==2 :
                        g[m][n]=color
                    elif MODE[i] ==3 :
                        b[m][n]=color
                m+=1
            n+=1

    produceimage(r,g,b,output_dir)

def MAINP(source_dir,output_dir,num_p,ratio_p,color_arr,sizepic_x,sizepic_y):

    os.mkdir(output_dir)

    ARR1,ARR2,ARR3=process_flight_data(source_dir,ratio_p)

    ARR1_new=getnewARR(ARR1,num_p,sizepic_x,sizepic_y)
    ARR2_new=getnewARR(ARR2,num_p,sizepic_x,sizepic_y)
    ARR3_new=getnewARR(ARR3,num_p,sizepic_x,sizepic_y)

    r = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    g = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    b = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)

    X1ALL,Y1ALL,X2ALL,Y2ALL,X3ALL,Y3ALL,MODE=[],[],[],[],[],[],[]
    mode=0

    for mx in ARR3_new:

        X1,Y1,X2,Y2,X3,Y3,mode=Left_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Right_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Tail_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Head_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3


    for i in range(len(X1ALL)):

        x1, y1, x2, y2, x3, y3=X1ALL[i],Y1ALL[i],X2ALL[i],Y2ALL[i],X3ALL[i],Y3ALL[i]
        max_x,min_x,max_y,min_y=getborder(x1, y1, x2, y2, x3, y3,sizepic_x,sizepic_y)
        color=color_arr[0]
        n=min_y
        m=min_x
        while(n<max_y):
            m=min_x
            while(m<max_x):
                if insidetri(m ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n+1 , x1, y1, x2, y2, x3, y3) or insidetri(m ,n+1 , x1, y1, x2, y2, x3, y3) :
                    if MODE[i] ==1 :
                        r[m][n]=color
                    elif MODE[i] ==2 :
                        g[m][n]=color
                    elif MODE[i] ==3 :
                        b[m][n]=color
                m+=1
            n+=1

    produceimage(r,g,b,os.path.join(output_dir,'03.png'))
    r = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    g = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    b = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    X1ALL,Y1ALL,X2ALL,Y2ALL,X3ALL,Y3ALL,MODE=[],[],[],[],[],[],[]
    mode=0

    for mx in ARR2_new:

        X1,Y1,X2,Y2,X3,Y3,mode=Left_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Right_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Tail_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Head_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

    for i in range(len(X1ALL)):

        x1, y1, x2, y2, x3, y3=X1ALL[i],Y1ALL[i],X2ALL[i],Y2ALL[i],X3ALL[i],Y3ALL[i]
        color=color_arr[1]
        max_x,min_x,max_y,min_y=getborder(x1, y1, x2, y2, x3, y3,sizepic_x,sizepic_y)
        n=min_y
        m=min_x
        while(n<max_y):
            m=min_x
            while(m<max_x):
                if insidetri(m ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n+1 , x1, y1, x2, y2, x3, y3) or insidetri(m ,n+1 , x1, y1, x2, y2, x3, y3) :
                    if MODE[i] ==1 :
                        r[m][n]=color
                    elif MODE[i] ==2 :
                        g[m][n]=color
                    elif MODE[i] ==3 :
                        b[m][n]=color
                m+=1
            n+=1

    produceimage(r,g,b,os.path.join(output_dir,'02.png'))
    r = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    g = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    b = np.zeros((sizepic_x, sizepic_y), dtype=np.uint8)
    X1ALL,Y1ALL,X2ALL,Y2ALL,X3ALL,Y3ALL,MODE=[],[],[],[],[],[],[]
    mode=0

    for mx in ARR1_new:

        X1,Y1,X2,Y2,X3,Y3,mode=Left_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Right_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Tail_Triangle_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

        X1,Y1,X2,Y2,X3,Y3,mode=Head_find(mx)
        MODE+=[mode for _ in X1]
        X1ALL+=X1
        X2ALL+=X2
        X3ALL+=X3
        Y1ALL+=Y1
        Y2ALL+=Y2
        Y3ALL+=Y3

    for i in range(len(X1ALL)):

        x1, y1, x2, y2, x3, y3=X1ALL[i],Y1ALL[i],X2ALL[i],Y2ALL[i],X3ALL[i],Y3ALL[i]
        color=color_arr[2]
        max_x,min_x,max_y,min_y=getborder(x1, y1, x2, y2, x3, y3,sizepic_x,sizepic_y)
        n=min_y
        m=min_x
        while(n<max_y):
            m=min_x
            while(m<max_x):
                if insidetri(m ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n , x1, y1, x2, y2, x3, y3) or insidetri(m+1 ,n+1 , x1, y1, x2, y2, x3, y3) or insidetri(m ,n+1 , x1, y1, x2, y2, x3, y3) :
                    if MODE[i] ==1 :
                        r[m][n]=color
                    elif MODE[i] ==2 :
                        g[m][n]=color
                    elif MODE[i] ==3 :
                        b[m][n]=color
                m+=1
            n+=1
    produceimage(r,g,b,os.path.join(output_dir,'01.png'))

# 主程序

if __name__ == "__main__":

    prcss_max_num=15#进程数量

    num_p=8     #每张图片飞机的个数
    ratio_p=0.075#飞机尺寸大小  最小万分之一0.0625
    mode=3     #选取几个平面

    #上色组
    color_arr=[42,127,213]

    #最好是方形
    sizepic_x=64#图片尺寸横
    sizepic_y=64#图片尺寸纵

    ifback=0#记忆（最好为0），是否返回
    ifmass=1#是否批量生成

    NUM_C=1000#限制单动作样本个数，即统一每个种类个数相等。0代表最大
    NUM_L=50#样本中的最低采样数的.
    NUM_A=0#种类，不用管

    #批量生成文件夹地址
    base_dir='C:\\000 onedrive 备份\\OneDrive\\桌面\\动作识别汇总(0_219)'
    output_dir='C:\\000 onedrive 备份\\OneDrive\\桌面'
    test_acin=0.2#验证的比例   最小百分之一

    #单次生成具体文件地址
    source_dir = "C:\\000 onedrive 备份\\OneDrive\\桌面\\sample01.txt"
    output_pic_dir='C:\\000 onedrive 备份\\OneDrive\\桌面\\test002'

    start_time=time.time()
    if ifmass:
        NUM_A=PREMAIN(NUM_C,NUM_L,ifback,base_dir)
        output_dir=os.path.join(output_dir,str(mode)+'figures'+'(C='+str(NUM_C)+'_L='+str(NUM_L)+'_A='+str(NUM_A)+'_X='+str(sizepic_x)+'_Y='+str(sizepic_y)+'_P='+str(num_p)+'_T='+str(round(test_acin*100))+'_O='+str(round(ratio_p*10000))+')')
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir,'train'))
        os.mkdir(os.path.join(output_dir,'test'))
        name01=os.listdir(base_dir)
        print('图片生成中...')
        progress_bar=tqdm(total=(NUM_A*NUM_C))
        for name01_t in name01 :
            if 'tempt' not in name01_t :
                os.mkdir(os.path.join(output_dir,'train',name01_t))
                os.mkdir(os.path.join(output_dir,'test',name01_t))
                name02=os.listdir(os.path.join(base_dir,name01_t))
                random_numbers = random.sample(range(len(name02)), NUM_C)
                source_dirlist=[]
                output_dirlist=[]
                gl_mincounter=0
                gl_maxcounter=0
                
                for i in random_numbers:
                    source_dirlist.append(os.path.join(base_dir,name01_t,name02[i]))
                    output_dirlist.append(os.path.join(output_dir,'train',name01_t,name02[i]+'.png'))

                while(gl_mincounter<NUM_C):

                    if (gl_mincounter+prcss_max_num)<NUM_C:
                        gl_maxcounter+=prcss_max_num
                    elif gl_mincounter<NUM_C:
                        gl_maxcounter=NUM_C

                    processes=[]
                    for i in range(gl_mincounter,gl_maxcounter):
                        p = multiprocessing.Process(target=MAIN, args=(source_dirlist[i],output_dirlist[i],num_p,ratio_p,color_arr,sizepic_x,sizepic_y,mode))
                        p.start()
                        processes.append(p)

                    for process in processes :
                        process.join()

                    progress_bar.update(gl_maxcounter-gl_mincounter)
                    gl_mincounter=gl_maxcounter

        progress_bar.close()



        print('测试集分离中...')
        progress_bar=tqdm(total=(round(test_acin*NUM_A*NUM_C)))
        name01=os.listdir(os.path.join(output_dir,'train'))
        for name01_t in name01 :
            name02=os.listdir(os.path.join(output_dir,'train',name01_t))
            random_numbers = random.sample(range(len(name02)), round(NUM_C*test_acin))
            for i in random_numbers:
                shutil.move(os.path.join(output_dir,'train',name01_t,name02[i]),os.path.join(output_dir,'test',name01_t,name02[i]))
                progress_bar.update(1)
        progress_bar.close()

        print('顺序命名图片中...')
        
        progress_bar=tqdm(total=(NUM_A*NUM_C))
        name01=os.listdir(os.path.join(output_dir,'train'))
        for name01_t in name01 :
            counter=0
            name02=os.listdir(os.path.join(output_dir,'train',name01_t))
            for name02_t in name02:
                counter+=1
                os.rename(os.path.join(output_dir,'train',name01_t,name02_t),os.path.join(output_dir,'train',name01_t,'0'*(numdigits(NUM_A*NUM_C)-numdigits(counter))+str(counter)+'.png'))
                progress_bar.update(1)

        name01=os.listdir(os.path.join(output_dir,'test'))
        for name01_t in name01 :
            counter=0
            name02=os.listdir(os.path.join(output_dir,'test',name01_t))
            for name02_t in name02:
                counter+=1
                os.rename(os.path.join(output_dir,'test',name01_t,name02_t),os.path.join(output_dir,'test',name01_t,'0'*(numdigits(NUM_A*NUM_C)-numdigits(counter))+str(counter)+'.png'))
                progress_bar.update(1)
        progress_bar.close()

        print('已生成'+str(NUM_A*NUM_C)+'个图片。 '+'总共耗费时长：'+str(round(time.time()-start_time))+' 秒。')
    
    else:
        MAINP(source_dir,output_pic_dir,num_p,ratio_p,color_arr,sizepic_x,sizepic_y)
        MAIN(source_dir,os.path.join(output_pic_dir,'sum'+'(C='+str(NUM_C)+'_L='+str(NUM_L)+'_A='+str(NUM_A)+'_X='+str(sizepic_x)+'_Y='+str(sizepic_y)+'_P='+str(num_p)+'.png'),num_p,ratio_p,color_arr,sizepic_x,sizepic_y,mode)
        print('已生成'+'1组图片。 '+'总共耗费时长：'+str(round(time.time()-start_time))+' 秒。')


    


