#定义 经度是longitude
#     纬度是latitude
#时间为一单独数组：time
import numpy as np
import math
import sys
from cllct_data_of_a_partclr_pln import requiredata

source_dir=''

sys.setrecursionlimit(1000000)
limit_pip_ll=0.05 #经纬阈值
limit_pip_hi=5 #高度阈值
pip_set_ll=[]
pip_set_hi=[]

c_ph=[] #高度上的改变点
x_ph=[] #高度上的改变方向
k_ll=[] #第i个到i+1的斜率(角)
c_ll=[] #
x_ll=[] #
def Init_pip():
    pip_set_hi.append(0)
    pip_set_hi.append(len(height)-1)
    pip_set_ll.append(0)
    pip_set_ll.append(len(longitude)-1)

time ,longitude ,latitude ,height ,roll ,pitch ,yaw =requiredata('305',source_dir)

# 添加高斯噪声


start=2550
end=2750
height=height[start:end]
longitude=longitude[start:end]
latitude=latitude[start:end]
time=time[start:end]


noise = np.random.normal(0, 3, len(height))  # 均值为0，标准差为0.5
height = height + noise#给高度加噪点

def uncentity1 (longitude,latitude,l_pip,r_pip):
        k=(longitude[l_pip]-longitude[r_pip])/(latitude[l_pip]-latitude[r_pip])
        b=longitude[l_pip]-k*latitude[l_pip]
        sum=0
        for j in range (l_pip,r_pip):
            sum+=(longitude[l_pip]-(k*latitude[j]+b))**2
        n=r_pip-l_pip+1
        res=np.sqrt(sum/n)
        return res


def find_lltude(longitude,latitude,l_pip,r_pip):
    
    k=(longitude[l_pip]-longitude[r_pip])/(latitude[l_pip]-latitude[r_pip])
    b=longitude[l_pip]-k*latitude[l_pip]
    linear=[]
    for i in range(l_pip+1,r_pip-1):
        linear.append(k*i+b)
   # linear=map(lambda x : k*x+b , [latitude[i] for i in range(l_pip+1,r_pip)])#在直线下的理想经度
    
    temp=0
    index=0
    
    for i in range(l_pip+1,r_pip-1):
        if temp<=abs(longitude[i]-linear[i-l_pip-1]):
            temp=abs(longitude[i]-linear[i-l_pip-1])#计算差值是否是最大的
            index=i#记录
    
    return index

def Deal_ll(l,r):  #经纬递归pip
    if l>=r:
        return
    lim1=uncentity1(longitude,latitude,l,r)
    
    if lim1<=limit_pip_ll:
        return
    mid=find_lltude(longitude,latitude,l,r)
    pip_set_ll.append(mid)
    print(mid)
    Deal_ll(l,mid)
    Deal_ll(mid+1,r)
    return

def uncentity (height,l_pip,r_pip):

    k=(height[l_pip]-height[r_pip])/(l_pip-r_pip)
    b=height[l_pip]-k*l_pip
    sum=0
    for j in range (l_pip,r_pip):
        sum+=(height[j]-(k*j+b))**2
    n=r_pip-l_pip+1
    res=np.sqrt(sum/n)
    return res

def find_pip1(height,l_pip,r_pip):
    
    k=(height[l_pip]-height[r_pip])/(l_pip-r_pip)
    b=height[l_pip]-k*l_pip
    linear=[]
    for i in range(l_pip+1,r_pip-1):
        linear.append(k*i+b)
    #linear=map(lambda x : k*x+b , [i for i in range(l_pip+1,r_pip)])#在直线下的理想高度
    temp=0
    index=l_pip
    
    for i in range(l_pip+1,r_pip-1):
        if temp<=abs(height[i]-linear[i-l_pip-1]):
            temp=abs(height[i]-linear[i-l_pip-1])#计算差值是否是最大的
            index=i#记录位置
    
    return index

def Deal_hi(l,r):  #高度递归pip
        
        if l>=r:
            return
        lim1=uncentity(height,l,r)
        if lim1<=limit_pip_hi:
            return
        mid=find_pip1(height,l,r)
        # print(height[mid])
        pip_set_hi.append(mid)
        Deal_hi(l,mid)
        Deal_hi(mid+1,r)
        return

    
def fly_hi():
    pre_pose=0
    #1升 0平 -1降
    for i in range(1,len(pip_set_hi)-1):
        now=height[pip_set_hi[i]]-height[pip_set_hi[i-1]] 
        #print(now)
        if (now>0):
            if (pre_pose==1):
                continue
            else:
                print("nmsl")
                pre_pose=1
                c_ph.append(pip_set_hi[i-1])
                x_ph.append(1)
        elif (now==0):
            if (pre_pose==0 ):
                continue
            else:
                pre_pose=0
                c_ph.append(pip_set_hi[i-1])
                x_ph.append(0)
        elif (now<0):
            if (pre_pose==-1):
                continue
            else:
                pre_pose=-1
                c_ph.append(pip_set_hi[i-1])
                x_ph.append(-1)
 #   c_ph.append(pip_set_hi[len(pip_set_hi)-1])
#    x_ph.append()
def fly_ll():
    pre_pose=0
    #-1 右 0 不变 1 左
    for i in range(0,len(pip_set_ll)-2):
        k_ll.append(math.atan2((longitude[pip_set_ll[i]]-longitude[pip_set_ll[i+1]]),(latitude[pip_set_ll[i]]-latitude[pip_set_ll[i+1]])))
    for i in range(0,len(k_ll)-2):
        now=k_ll[i+1]-k_ll[i]
        if (now>0):
            if (pre_pose>0):
                continue
            else:
                pre_pose=1
                c_ll.append(pip_set_ll[i])
                x_ll.append(1)
        elif (now==0):
            if (pre_pose==0):
                continue
            else:
                pre_pose=0
                c_ll.append(pip_set_ll[i])
                x_ll.append(0)
        elif (now<0):
            if (pre_pose==-1):
                continue
            else:
                pre_pose=-1
                c_ll.append(pip_set_ll[i])
                x_ll.append(-1)
    
def Solve_result():
    ind_hi=-1
    ind_ll=-1
    now_hi=0
    now_ll=0
    past_time=0
    now_time=0
    pack1=open("C:\\Users\\Poter\\Desktop\\数据识别.txt","wt")
    while (ind_hi<len(c_ph)-1 and ind_ll<len(c_ll)-1):
        if (time[c_ph[ind_hi+1]]<time[c_ll[ind_ll+1]]) :
            ind_hi+=1
            now_time=time[c_ph[ind_hi]]
            now_hi=x_ph[ind_hi]
        elif (time[c_ph[ind_hi+1]]>time[c_ll[ind_ll+1]]) :
            ind_ll+=1
            now_ll=x_ll[ind_ll]
            now_time=time[c_ll[ind_ll]]
        else :
            ind_hi+=1
            ind_ll+=1
            now_hi=x_ph[ind_hi]
            now_ll=x_ll[ind_ll]
            now_time=time[c_ph[ind_hi]]
        pack1.write("从"+str(past_time)+"到"+str(now_time)+"的时间段内，飞机的飞行趋势为")
        past_time=now_time
        if(now_hi>0):#装呗给我飞起来    
            if(now_ll>0):
                pack1.write("向左爬升")
            elif(now_ll==0):
                pack1.write("爬升")
            else:
                pack1.write("向右爬升")
        if(now_hi==0):#平   
            if(now_ll>0):
                pack1.write("向左平飞")
            elif(now_ll==0):
                pack1.write("平飞")
            else:
                pack1.write("向右平飞")
        if(now_hi<0):#平   
            if(now_ll>0):
                pack1.write("向左下降")
            elif(now_ll==0):
                pack1.write("下降")
            else:
                pack1.write("向右下降")  
        pack1.write("\n") 
    pack1.write("从"+str(past_time)+"到"+str(now_time)+"的时间段内，飞机的飞行趋势为")
    if(now_hi>0):#装呗给我飞起来    
        if(now_ll>0):
            pack1.write("向左爬升")
        elif(now_ll==0):
            pack1.write("爬升")
        else:
            pack1.write("向右爬升")
    if(now_hi==0):#平   
        if(now_ll>0):
            pack1.write("向左平飞")
        elif(now_ll==0):
            pack1.write("平飞")
        else:
            pack1.write("向右平飞")
    if(now_hi<0):#平   
        if(now_ll>0):
            pack1.write("向左下降")
        elif(now_ll==0):
            pack1.write("下降")
        else:
            pack1.write("向右下降")
    pack1.write("\n")
    pack1.close()      
    
def printphoto_pip (height,time,pip_set_hi):

    
    time=[time[pip_set_hi[i]] for i in range(len(pip_set_hi))]
    height=[height[pip_set_hi[i]] for i in range(len(pip_set_hi))]

    print(time)
    print(height)

    import matplotlib.pyplot as plt
    plt.close()
    # 假设 height 是一个已经存在的列表，包含了高度数据
    # 生成与 height 列表长度相同的下标序列

    plt.figure(figsize=(15, 8))
    plt.scatter(time, height, color='blue', marker='o')
    plt.title('delta='+str(limit_pip_hi)+"and"+str(limit_pip_ll)+"       Change of altitude over time")
    #plt.title('Change of altitude over time')    
    plt.xlabel('time(second)')
    plt.ylabel('height(meter)')
    plt.grid(True)
    plt.show()

    return None


def Main():

    Init_pip()
    Deal_hi(0,len(height)-1)
    Deal_ll(0,len(longitude)-1)
    pip_set_hi.sort()
    pip_set_ll.sort()
    for i in range(0,len(pip_set_hi)):
       print(height[pip_set_hi[i]])
    fly_hi()
    fly_ll()
    print(len(pip_set_hi))
    print(len(pip_set_ll))
    printphoto_pip (height,time,pip_set_hi)
    Solve_result()


Main()
