from collectdata02 import RQDATA
from tqdm import tqdm
import numpy as np
import random
import shutil
import time
import math
import os

source_dir = 'C:\\000 onedrive 备份\\OneDrive\\桌面\\动作识别汇总(0_219)'
output_dir = 'C:\\000 onedrive 备份\\OneDrive\\桌面\\神经网络angles训练数据'

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

def anglenise (source_ptclrdir, NUM_F) :
    earth_radius = 6371
    longitude ,latitude ,height ,roll ,pitch ,yaw , _=RQDATA(source_ptclrdir)

    longitude=[np.float32(item) for item in longitude]
    latitude=[np.float32(item) for item in latitude]
    height=[np.float32(item) for item in height]
    roll=[np.float32(item) for item in roll]
    pitch=[np.float32(item) for item in pitch]
    yaw=[np.float32(item) for item in yaw]   

    step=len(longitude)//(NUM_F)
    PIPARR=[0+step*i for i in range(NUM_F+1)]
    PIPARR[NUM_F]=len(longitude)-1

    arr = np.zeros((NUM_F, 6))

    for i in range(NUM_F) :
        x=earth_radius * math.cos(math.radians((longitude[PIPARR[i+1]]))) * math.cos(math.radians((latitude[PIPARR[i+1]])))-earth_radius * math.cos(math.radians((longitude[PIPARR[i]]))) * math.cos(math.radians((latitude[PIPARR[i]])))
        y=earth_radius * math.sin(math.radians((longitude[PIPARR[i+1]]))) * math.cos(math.radians((latitude[PIPARR[i+1]])))-earth_radius * math.sin(math.radians((longitude[PIPARR[i]]))) * math.cos(math.radians((latitude[PIPARR[i]])))
        z=(height[PIPARR[i+1]])-(height[PIPARR[i]])

        sqrtval=math.sqrt(x**2+y**2+z**2)
        if sqrtval ==0 :
            sqrtval=1

        angle_x=round(math.degrees(math.acos(x*1/sqrtval)),1)
        angle_y=round(math.degrees(math.acos(y*1/sqrtval)),1)
        angle_z=round(math.degrees(math.acos(z*1/sqrtval)),1)

        arr[i][0]=angle_x
        arr[i][1]=angle_y
        arr[i][2]=angle_z
        arr[i][3]=(roll[round((PIPARR[i]+PIPARR[i+1])/2)])
        arr[i][4]=(pitch[round((PIPARR[i]+PIPARR[i+1])/2)])
        arr[i][5]=(yaw[round((PIPARR[i]+PIPARR[i+1])/2)])

    return arr

def dirget (source_dir,NUM_C):

    name01=[i for i in os.listdir(source_dir) if i != ('temptLis'+str(NUM_L))]
    labelbase = np.array([name01_v for name01_v in name01], dtype='U')
    amount=[0]*len(labelbase)
    counter=0

    for name01_v in name01 :
        amount[counter]=len(os.listdir(os.path.join(source_dir,name01_v)))
        counter+=1

    for i in amount :
        if i < NUM_C :
            NUM_C=i

    random_array=[]
    for i in range(len(amount)) :
        random_tempt=random.sample(range(amount[i]),NUM_C)
        random_array+=random_tempt

    r_l=[]
    for i in range(len(labelbase)):
        r_l+=[i]*NUM_C

    random.shuffle(r_l)

    label=np.array(r_l)

    counter_tempt=[0 for _ in name01]
    totalpath=[]
    for i in range(len(label)):
        totalpath.append(os.path.join(name01[label[i]],os.listdir(os.path.join(source_dir,name01[label[i]]))[random_array[r_l[i]*NUM_C+counter_tempt[r_l[i]]]]))
        counter_tempt[r_l[i]]+=1

    return totalpath ,label

def MAIN (NUM_C,NUM_F,NUM_L,ifdel,ifback,source_dir,output_dir,ifmass):
    if ifmass==0:
        start_time=time.time()
        ifdel=0
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
    if ifmass==0:
        print('允许设置 NUM_C 的最大值：'+str(nummin))
    if NUM_C>nummin or NUM_C==0:
        NUM_C=nummin

    totalpath, label=dirget(source_dir, NUM_C)
    array_list = []
    for i in range(len(totalpath)) :
        source_ptclrdir=os.path.join(source_dir,totalpath[i])    
        arr=anglenise(source_ptclrdir, NUM_F)
        array_list.append(arr)

    datangltype=np.array(array_list)

    if ifdel :
        cns=input("是否删除(1代表是0代表否):")
        if int(cns)==1:
            shutil.rmtree(os.path.join(source_dir,'temptLis'+str(NUM_L)))
        elif ifback:
            endback02(source_dir,NUM_L)
    elif ifback:
        endback02(source_dir,NUM_L)

    np.savez_compressed(os.path.join(output_dir,'data'+'(C='+str(NUM_C)+'_F='+str(NUM_F)+'_L='+str(NUM_L)+'_A='+str(NUM_A)+')'+'.npz'), dtrn=datangltype, lbltrn=label)
    if ifmass==0:
        print('已完成。'+' '+'总共耗费时长：'+str(round(time.time()-start_time))+' 秒。')

NUM_C=100#限制单动作样本个数，即统一每个种类个数相等。0代表最大
NUM_F=10#限制单样本的采样深度。
NUM_L=50#样本中的最低采样数的

ifdel=0#是否删除
ifback=0#记忆（最好为0），是否返回
ifmass=0#是否批量生成

if ifmass :
    NUM_Carr=[50,100,150,200,250,300,350,400,450,500,550,600,650,700]
    NUM_FdL=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    progress_bar=tqdm(total=len(NUM_Carr)*len(NUM_FdL))
    start_time=time.time()
    for m in NUM_Carr:
        NUM_C=m
        for n in NUM_FdL:
            NUM_F=int(NUM_L*n)
            MAIN(NUM_C,NUM_F,NUM_L,ifdel,ifback,source_dir,output_dir,ifmass)
            progress_bar.update(1)
    progress_bar.close()
    print('已生成'+str(len(NUM_Carr)*len(NUM_FdL))+'个数据文件。 '+'总共耗费时长：'+str(round(time.time()-start_time))+' 秒。')
else:
    MAIN(NUM_C,NUM_F,NUM_L,ifdel,ifback,source_dir,output_dir,ifmass)





