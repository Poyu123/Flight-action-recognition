# 时间数组time
# 纬度数组latitude
# 经度数组longitude
# 高度数组height
# 翻滚角度roll
# 俯仰角  pitch
# 偏航角  Yaw
# 修改日期：2024/8/24/22/00/00
import math
# import plotly.graph_objs as go
import numpy as np
from tkinter import filedialog
import os
import gc
import time
from datetime import datetime
from cllct_data_of_a_partclr_pln import requiredata

global source_dir
global name_dir
global output_dir

def select_folder():
    folder_selected=''
    folder_selected = filedialog.askopenfilename(filetypes=[("文本文件", "*.txt")])
    return folder_selected

print("请选择数据来源：")
source_dir=''
while not source_dir :
    source_dir=select_folder()
print('已选择'+source_dir+"作为数据来源")
print("请选择花名册来源：")
name_dir=''
while not name_dir :
    name_dir=select_folder()
print('已选择'+name_dir+"作为花名册来源")
print("请选择输出地址：")
output_dir=''
while not output_dir :
    output_dir = filedialog.askdirectory()
output_dir=os.path.join(output_dir,os.path.basename(os.path.dirname(source_dir))[:3]+'_横滚识别'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
print('已选择'+output_dir+"作为输出地址")

'''
source_dir='C:\\000 onedrive 备份\\OneDrive\\桌面\\飞机数据\\Tacview_data.acmi\\飞行数据.txt'
name_dir='C:\\000 onedrive 备份\\OneDrive\\桌面\\飞机数据\\Tacview_data.acmi\\飞机编号.txt'
output_dir='C:\\000 onedrive 备份\\OneDrive\\桌面'
output_dir=os.path.join(output_dir,'横滚识别'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
'''

def rolldelta (R1,R2):

    deltaR=0

    if R1*R2 >= 0 :#同号
        deltaR=R2-R1
    elif abs(R1-R2)<=180 :#过0线
        deltaR=R2-R1
    else :#过180线
        deltaR=abs(360-abs(R1)-abs(R2))
        if R1<0 or R2>0 :
            deltaR*=(-1)
    
    return deltaR   

def printphoto(height,time,output_dir,start,end,ID):

    import matplotlib.pyplot as plt
    height=height[start:end]
    time=time[start:end]
    name=plt.figure(figsize=(15, 8))
    plt.scatter(time, height, color='blue', marker='o')
    plt.title('Change of roll over time')
    plt.xlabel('time')
    plt.ylabel('roll')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, str(ID)+'_'+str(time[0])+'_to_'+str(time[-1])+'.png'))
    plt.close
    del plt
    del name
    gc.collect()
    return None

def fullroll(ID):
    os.mkdir(os.path.join(output_dir,ID))
    time, longitude, latitude, height, roll, pitch, yaw=requiredata(ID,source_dir)
    vary_num=30#一个周期的度数变化范围
    trigger_num=3#连续单调个数
    variance_num=5#变化数组方差限制
    counter_num=0#横滚片段个数
    last_distance=360
    max_num=len(roll)-1
    trigger_counter=0
    counter=0

    while(1):
        while(1):#trigger 寻找trigger_num个点都是单增的
            if (trigger_counter+trigger_num) < max_num :
                sign=1
                tempt_num=0
                for i in range(trigger_counter,(trigger_counter+trigger_num-1)):
                    if rolldelta(roll[i],roll[i+1]) :
                        sign*=rolldelta(roll[i],roll[i+1])
                        tempt_num+=1
                    if sign <0 :
                        break
                if sign > 0 :
                    #print(sign)
                    #print("识别到可能触发点")
                    break
                else:
                    #print('2')
                    trigger_counter+=tempt_num
            else:
                break

        #print(str(trigger_counter+trigger_num)+'@@'+str(max_num))
        if (trigger_counter+trigger_num) >= max_num :
            break

        #print(3)
        counter=trigger_counter
        sum_roll=rolldelta(roll[counter],roll[counter+1])

        while(1):
            if ((counter+2)>=max_num) :
                break
            delta_1=rolldelta(roll[counter],roll[counter+1])
            delta_2=rolldelta(roll[counter+1],roll[counter+2])
            if delta_1!=0 and delta_2!=0:
                if delta_1*delta_2 >0 :
                    sum_roll+=rolldelta(roll[counter+1],roll[counter+2])
                    counter+=1
                    #print(str(counter+2)+'@@'+str(max_num))
                    if (abs(abs(sum_roll)-360) <= vary_num) :
                        if ((counter+2)>=max_num) :
                            break
                        if  last_distance > abs(abs(sum_roll)-360) :
                            last_distance=abs(abs(sum_roll)-360)
                        else :
                            counter-=1
                            break
                else:
                    break
            else:
                if delta_1==0 :
                    counter+=1
                    if ((counter+2)>=max_num):
                        break
                elif delta_2==0 :
                    i=counter+1
                    while(rolldelta(roll[i],roll[i+1])==0):
                        i+=1
                        if ((i+1)>=max_num):
                            break
                    delta_2=rolldelta(roll[i],roll[i+1])
                    counter=i-1
                    if ((counter+2)>=max_num):
                        break

        if abs(sum_roll-360) <= vary_num :
            deltaR=[0.0]*(counter-trigger_counter+1)
            counter=trigger_counter
            for i in range(0,len(deltaR)) :
                #print(rolldelta(roll[counter],roll[counter+1]))
                deltaR[i]=rolldelta(roll[counter],roll[counter+1])
                counter+=1
            mean = sum(deltaR) / len(deltaR)
            variance = math.sqrt(sum((x - mean) ** 2 for x in deltaR) / len(deltaR))
            if variance <= variance_num :
                #print(deltaR)
                print("已寻找到符合条件的飞行区段")
                print("正在写入...")
                with open(os.path.join(output_dir,ID,(str(time[trigger_counter])+'_to_'+str(time[counter])+'.txt')), 'w') as file:
                    for i in range(trigger_counter,(counter+1)):
                        file.write(str(longitude[i])+' | '+str(latitude[i])+' | '+str(height[i])+' | '+str(roll[i])+' | '+str(pitch[i])+' | '+str(yaw[i])+' | '+str(time[i])+"\n")
                print("写入完成...")
                counter_num=counter_num+1
                printphoto(roll,time,os.path.join(output_dir,'photo'),trigger_counter,(counter+1),ID)

        trigger_counter=counter

    print("以完成对飞机号为"+str(ID)+"的横滚飞行动作的识别")
    print("有 "+str(counter_num)+" 段飞行符合条件")
    if counter_num==0 :
        os.rmdir(os.path.join(output_dir,ID))

start_time = time.time() 

with open(name_dir, 'r') as file:
    lines = file.readlines()
ID_all = [line.strip() for line in lines]
ID_all = list(dict.fromkeys(ID_all))
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir,'photo'))
counter=0
for id in ID_all :
    fullroll(id)
    counter=counter+1
    end_time = time.time()
    print("进度<--"+str(round(counter*100/len(ID_all),2))+"%-->"+"\t预计还需要: "+str(round((end_time - start_time)/(counter/len(ID_all))*(1-counter/len(ID_all))))+" 秒")

end_time = time.time()
print("程序运行时间: "+str(round(end_time - start_time))+" 秒")