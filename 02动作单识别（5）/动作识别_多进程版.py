# 修改日期：2024/8/26/22/00/00

from cllct_data_of_a_partclr_pln import requiredata
import multiprocessing
import matplotlib.pyplot as plt
from tkinter import filedialog
import time
import math
import os
import gc

def get_lotid(ID_all,prcss_max_num,gl_counter):
    if len(ID_all[gl_counter:])>=prcss_max_num :
        prcss_idlist=[ID_all[gl_counter+i] for i in range(prcss_max_num)]
        gl_counter=gl_counter+prcss_max_num
    else :
        prcss_idlist=[ID_all[gl_counter+i] for i in range(len(ID_all[gl_counter:]))]
        gl_counter=gl_counter+len(prcss_idlist)
    return prcss_idlist, gl_counter

def printphoto(height,time,output_dir,start,end,ID,mode=1):

    import matplotlib.pyplot as plt
    height=height[start:end]
    time=time[start:end]
    name=plt.figure(figsize=(15, 8))
    plt.scatter(time, height, color='blue', marker='o')
    
    if mode==1 :
        plt.title('Change of roll over time')
        plt.ylabel('roll')
    elif mode==2:
        plt.title('Change of pitch over time')
        plt.ylabel('pitch')
    elif mode==3:
        plt.title('Change of yaw over time')
        plt.ylabel('yaw')

    plt.xlabel('time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, str(ID)+'_'+str(time[0])+'_to_'+str(time[-1])+'.png'))
    plt.close
    del plt
    del name
    gc.collect()
    return None

def pitchdelta (R1,R2):

    deltaR=0

    if abs(R1-R2)<=180 or R1*R2 >= 0 :#同号#过0线
        deltaR=R2-R1
    else :#过180线
        deltaR=abs(360-abs(R1)-abs(R2))
        if R1<0 or R2>0 :
            deltaR*=(-1)
    
    return deltaR   

def rolldelta (R1,R2):

    deltaR=0

    if abs(R1-R2)<=180 or R1*R2 >= 0 :#同号#过0线
        deltaR=R2-R1
    else :#过180线
        deltaR=abs(360-abs(R1)-abs(R2))
        if R1<0 or R2>0 :
            deltaR*=(-1)
    
    return deltaR   

def yawdelta (R1,R2):
    deltaR=R2-R1

    if abs(deltaR)>=180 :
        if R1>R2 :
            deltaR+=360
        else:
            deltaR-=360
              
    return deltaR   

def calculate_average_pitch(pitch, start_index, end_index):
    total_pitch = sum(pitch[start_index:end_index + 1])
    average_pitch = total_pitch / (end_index - start_index + 1)
    return average_pitch

def fullroll(ID,source_dir,output_dir):
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
                    break
                else:
                    trigger_counter+=tempt_num
            else:
                break

        if (trigger_counter+trigger_num) >= max_num :
            break

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
                deltaR[i]=rolldelta(roll[counter],roll[counter+1])
                counter+=1
            mean = sum(deltaR) / len(deltaR)
            variance = math.sqrt(sum((x - mean) ** 2 for x in deltaR) / len(deltaR))
            if variance <= variance_num :
                with open(os.path.join(output_dir,ID,(str(time[trigger_counter])+'_to_'+str(time[counter])+'.txt')), 'w') as file:
                    for i in range(trigger_counter,(counter+1)):
                        file.write(str(longitude[i])+' | '+str(latitude[i])+' | '+str(height[i])+' | '+str(roll[i])+' | '+str(pitch[i])+' | '+str(yaw[i])+' | '+str(time[i])+"\n")
                counter_num=counter_num+1
                printphoto(roll,time,os.path.join(output_dir,'photo'),trigger_counter,(counter+1),ID,mode=1)

        trigger_counter=counter

    if counter_num==0 :
        os.rmdir(os.path.join(output_dir,ID))

def fullpitch(ID,source_dir,output_dir):
    os.mkdir(os.path.join(output_dir,ID))
    time, longitude, latitude, height, roll, pitch, yaw=requiredata(ID,source_dir)
    vary_num=90#一个周期的度数变化范围
    trigger_num=3#连续单调个数
    variance_num=10#变化数组方差限制
    counter_num=0#筋斗片段个数
    last_distance=360
    max_num=len(pitch)-1
    trigger_counter=0
    counter=0

    while(1):
        while(1):#trigger 寻找trigger_num个点都是单增的
            if (trigger_counter+trigger_num) < max_num :
                sign=1
                tempt_num=0
                for i in range(trigger_counter,(trigger_counter+trigger_num-1)):
                    if pitchdelta(pitch[i],pitch[i+1]) :
                        sign*=pitchdelta(pitch[i],pitch[i+1])
                        tempt_num+=1
                    if sign <0 :
                        break
                if sign > 0 :
                    break
                else:
                    trigger_counter+=tempt_num
            else:
                break

        if (trigger_counter+trigger_num) >= max_num :
            break

        counter=trigger_counter
        sum_pitch=pitchdelta(pitch[counter],pitch[counter+1])
        while(1):
            if ((counter+2)>=max_num) :
                break
            delta_1=pitchdelta(pitch[counter],pitch[counter+1])
            delta_2=pitchdelta(pitch[counter+1],pitch[counter+2])
            if delta_1!=0 and delta_2!=0:
                if delta_1*delta_2 >0 :
                    sum_pitch+=pitchdelta(pitch[counter+1],pitch[counter+2])
                    counter+=1
                    if (abs(abs(sum_pitch)-360) <= vary_num) :
                        if ((counter+2)>=max_num) :
                            break
                        if  last_distance > abs(abs(sum_pitch)-360) :
                            last_distance=abs(abs(sum_pitch)-360)
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
                    while(pitchdelta(pitch[i],pitch[i+1])==0):
                        i+=1
                        if ((i+1)>=max_num):
                            break
                    delta_2=pitchdelta(pitch[i],pitch[i+1])
                    counter=i-1
                    if ((counter+2)>=max_num):
                        break

        if abs(sum_pitch-360) <= vary_num :
            deltaR=[0.0]*(counter-trigger_counter+1)
            counter=trigger_counter
            for i in range(0,len(deltaR)) :
                deltaR[i]=pitchdelta(pitch[counter],pitch[counter+1])
                counter+=1
            mean = sum(deltaR) / len(deltaR)
            variance = math.sqrt(sum((x - mean) ** 2 for x in deltaR) / len(deltaR))
            if variance <= variance_num :
                with open(os.path.join(output_dir,ID,(str(time[trigger_counter])+'_to_'+str(time[counter])+'.txt')), 'w') as file:
                    for i in range(trigger_counter,(counter+1)):
                        file.write(str(longitude[i])+' | '+str(latitude[i])+' | '+str(height[i])+' | '+str(roll[i])+' | '+str(pitch[i])+' | '+str(yaw[i])+' | '+str(time[i])+"\n")
                counter_num=counter_num+1
                printphoto(pitch,time,os.path.join(output_dir,'photo'),trigger_counter,(counter+1),ID,mode=2)

        trigger_counter=counter

    if counter_num==0 :
        os.rmdir(os.path.join(output_dir,ID))

def fullyaw(ID,source_dir,output_dir):
    os.mkdir(os.path.join(output_dir,ID))
    time, longitude, latitude, height, roll, pitch, yaw=requiredata(ID,source_dir)
    vary_num=10#一个周期的度数变化范围
    trigger_num=5#连续单调个数
    variance_num=10#变化数组方差限制
    counter_num=0#转圈片段个数
    last_distance=360
    max_num=len(yaw)-1
    trigger_counter=0
    counter=0

    while(1):

        if (trigger_counter+trigger_num) >= max_num :
            break
        while(1):#trigger 寻找trigger_num个点都是单增的
            if (trigger_counter+trigger_num) < max_num :
                sign=1
                tempt_num=0
                for i in range(trigger_counter,(trigger_counter+trigger_num-1)):
                    if yawdelta(yaw[i],yaw[i+1]) :
                        sign*=yawdelta(yaw[i],yaw[i+1])
                        tempt_num+=1
                    if sign <0 :
                        break
                if sign > 0 :
                    break
                else:
                    trigger_counter+=tempt_num
            else:
                break
        if (trigger_counter+trigger_num) >= max_num :
            break
        counter=trigger_counter
        sum_yaw=yawdelta(yaw[counter],yaw[counter+1])
        while(1):
            if ((counter+2)>=max_num) :
                break
            delta_1=yawdelta(yaw[counter],yaw[counter+1])
            delta_2=yawdelta(yaw[counter+1],yaw[counter+2])
            if delta_1!=0 and delta_2!=0:
                if delta_1*delta_2 >0 :
                    sum_yaw+=yawdelta(yaw[counter+1],yaw[counter+2])
                    counter+=1
                    if (abs(abs(sum_yaw)-360) <= vary_num) :
                        if ((counter+2)>=max_num) :
                            break
                        if  last_distance > abs(abs(sum_yaw)-360) :
                            last_distance=abs(abs(sum_yaw)-360)
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
                    while(yawdelta(yaw[i],yaw[i+1])==0):
                        i+=1
                        if ((i+1)>=max_num):
                            break
                    delta_2=yawdelta(yaw[i],yaw[i+1])
                    counter=i-1
                    if ((counter+2)>=max_num):
                        break

        if abs(sum_yaw-360) <= vary_num :
            deltaR=[0.0]*(counter-trigger_counter+1)
            counter=trigger_counter
            for i in range(0,len(deltaR)) :
                deltaR[i]=yawdelta(yaw[counter],yaw[counter+1])
                counter+=1
            mean = sum(deltaR) / len(deltaR)
            variance = math.sqrt(sum((x - mean) ** 2 for x in deltaR) / len(deltaR))
            if variance <= variance_num :
                with open(os.path.join(output_dir,ID,(str(time[trigger_counter])+'_to_'+str(time[counter])+'.txt')), 'w') as file:
                    for i in range(trigger_counter,(counter+1)):
                        file.write(str(longitude[i])+' | '+str(latitude[i])+' | '+str(height[i])+' | '+str(roll[i])+' | '+str(pitch[i])+' | '+str(yaw[i])+' | '+str(time[i])+"\n")
                counter_num=counter_num+1
                printphoto(yaw,time,os.path.join(output_dir,'photo'),trigger_counter,(counter+1),ID,mode=3)

        trigger_counter=counter

    if counter_num==0 :
        os.rmdir(os.path.join(output_dir,ID))

def identify_rapid_climbs_and_descents(ID, source_dir, output_dir):
    # 确保输出目录存在
    output_path = os.path.join(output_dir, ID)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    time, longitude, latitude, height, roll, pitch, yaw = requiredata(ID, source_dir)

    # 定义上升和下降的阈值
    threshold_pitch_ascent = 20  # 快速上升的俯仰角阈值
    threshold_rate_ascent = 50  # 快速上升的高度变化率阈值
    threshold_pitch_descent = -20  # 快速下降的俯仰角阈值
    threshold_rate_descent = -50  # 快速下降的高度变化率阈值

    # 初始化爬升和下降的列表
    rapid_climbs_details = []
    rapid_descents_details = []

    # 初始化爬升和下降的开始索引
    climb_start = None
    descent_start = None

    for i in range(1, len(height)):
        delta_height = height[i] - height[i - 1]
        delta_time = time[i] - time[i - 1]

        if delta_time == 0:
            continue  # 避免除以零

        height_rate = delta_height / delta_time

        # 检测快速上升
        if (pitch[i] > threshold_pitch_ascent and height_rate > threshold_rate_ascent):
            if climb_start is None:
                climb_start = i
        # 检测快速下降
        elif (pitch[i] < threshold_pitch_descent and height_rate < threshold_rate_descent):
            if descent_start is None:
                descent_start = i

        # 结束快速上升检测
        if climb_start is not None and (pitch[i] <= threshold_pitch_ascent or height_rate <= threshold_rate_ascent):
            climb_end = i - 1
            average_pitch = calculate_average_pitch(pitch, climb_start, climb_end)
            rapid_climbs_details.append({
                'start_time': time[climb_start],
                'end_time': time[climb_end],
                'average_pitch': average_pitch,
                'start_height': height[climb_start],
                'end_height': height[climb_end]
            })
            if(climb_start !=None and time[climb_end]-time[climb_start]>0):
                with open(os.path.join(output_dir, ID, (str(time[climb_start]) + '_to_' + str(time[climb_end]) + '.txt')),
                          'w') as file:
                    for i in range(climb_start, climb_end + 1):
                        file.write(str(longitude[i]) + ' | ' + str(latitude[i]) + ' | ' + str(height[i]) + ' | ' + str(
                            roll[i]) + ' | ' + str(pitch[i]) + ' | ' + str(yaw[i]) + ' | ' + str(time[i]) + "\n")
            climb_start = None

        # 结束快速下降检测
        if descent_start is not None and (pitch[i] >= threshold_pitch_descent or height_rate >= threshold_rate_descent):
            descent_end = i - 1
            average_pitch = calculate_average_pitch(pitch, descent_start, descent_end)
            rapid_descents_details.append({
                'start_time': time[descent_start],
                'end_time': time[descent_end],
                'average_pitch': average_pitch,
                'start_height': height[descent_start],
                'end_height': height[descent_end]
            })
            if(descent_start !=None and time[descent_end]-time[descent_start]>0):
                with open(os.path.join(output_dir, ID, (str(time[descent_start]) + '_to_' + str(time[descent_end]) + '.txt')),
                          'w') as file:
                    for i in range(descent_start, descent_end + 1):
                        file.write(str(longitude[i]) + ' | ' + str(latitude[i]) + ' | ' + str(height[i]) + ' | ' + str(
                            roll[i]) + ' | ' + str(pitch[i]) + ' | ' + str(yaw[i]) + ' | ' + str(time[i]) + "\n")
            descent_start = None




    # 处理最后一个上升或下降事件
    if climb_start is not None:
        rapid_climbs_details.append({
            'start_time': time[climb_start],
            'end_time': time[-1],
            'average_pitch': calculate_average_pitch(pitch, climb_start, len(height) - 1),
            'start_height': height[climb_start],
            'end_height': height[-1]
        })
        with open(os.path.join(output_dir, ID, (str(time[climb_start]) + '_to_' + str(time[-1]) + '.txt')),
                  'w') as file:
            for i in range(climb_start, len(time)):
                file.write(str(longitude[i]) + ' | ' + str(latitude[i]) + ' | ' + str(height[i]) + ' | ' + str(
                    roll[i]) + ' | ' + str(pitch[i]) + ' | ' + str(yaw[i]) + ' | ' + str(time[i]) + "\n")


    if descent_start is not None:
        rapid_descents_details.append({
            'start_time': time[descent_start],
            'end_time': time[-1],
            'average_pitch': calculate_average_pitch(pitch, descent_start, len(height) - 1),
            'start_height': height[descent_start],
            'end_height': height[-1]
        })
        with open(os.path.join(output_dir, ID, (str(time[descent_start]) + '_to_' + str(time[-1]) + '.txt')),
                  'w') as file:
            for i in range(descent_start, len(time)):
                file.write(str(longitude[i]) + ' | ' + str(latitude[i]) + ' | ' + str(height[i]) + ' | ' + str(
                    roll[i]) + ' | ' + str(pitch[i]) + ' | ' + str(yaw[i]) + ' | ' + str(time[i]) + "\n")



    # 记录快速上升和下降的详细信息
    if rapid_climbs_details or rapid_descents_details:
        with open(os.path.join(output_path, 'log.txt'), 'w') as file:
            for climb in rapid_climbs_details:
                if (climb['end_time'] - climb['start_time']) > 0:
                 file.write("快速上升事件：\n")
                 file.write("开始时间：" + str(climb['start_time']) + "\n")
                 file.write("结束时间：" + str(climb['end_time']) + "\n")
                 file.write("平均俯仰角：" + str(round(climb['average_pitch'])) + "度\n")
                 file.write("开始高度：" + str(climb['start_height']) + "m\n")
                 file.write("结束高度：" + str(climb['end_height']) + "m\n")
                 file.write("平均爬升速度：" + str(
                    round((climb['end_height'] - climb['start_height']) / (climb['end_time'] - climb['start_time']),
                          3)) + "m/s\n\n")

            for descent in rapid_descents_details:
                if (descent['end_time'] - descent['start_time'])>0:
                    file.write("快速下降事件：\n")
                    file.write("开始时间：" + str(descent['start_time']) + "\n")
                    file.write("结束时间：" + str(descent['end_time']) + "\n")
                    file.write("平均俯仰角：" + str(round(descent['average_pitch'])) + "度\n")
                    file.write("开始高度：" + str(descent['start_height']) + "m\n")
                    file.write("结束高度：" + str(descent['end_height']) + "m\n")
                    file.write("平均下降速度：" + str(round(
                        (descent['start_height'] - descent['end_height']) / (descent['end_time'] - descent['start_time']),
                        3)) + "m/s\n\n")

    else:
        os.rmdir(output_path)

if __name__ == '__main__':

    #最大允许运行的进程数量::
    prcss_max_num=-1    #0代表最大，-1代表半

    cnt_num=0#限制读取数据个数。！0则不作限制-

    iffullroll=1#是否识别横滚
    iffullpitch=1#是否识别筋斗
    iffullyaw=1#是否识别转圈
    ifrapidH=1#是否识别高度

    ifnotexist=[iffullpitch,iffullroll,iffullyaw,ifrapidH]#判断是否存在

    
    print("请选择数据来源：")
    base_dir=''
    while not base_dir :
        base_dir=filedialog.askdirectory()
    print('已选择'+base_dir+"作为数据来源")
    print("请选择输出地址：")
    output_dir=''
    while not output_dir :
        output_dir = filedialog.askdirectory()
    print('已选择'+output_dir+"作为输出地址")
    '''

    base_dir="C:\\000 onedrive 备份\\OneDrive\\桌面\\飞机数据"
    output_dir="C:\\000 onedrive 备份\\OneDrive\\桌面\\测试文件夹"
    '''
    name_ndo=[i for i in os.listdir(base_dir) if i not in os.listdir(output_dir) and i[0].isdigit()]
    name_nt=[i for i in os.listdir(base_dir) if i in os.listdir(output_dir) and i[0].isdigit()]

    for n_i in name_nt :
        if ('横滚动作识别' not in os.listdir(os.path.join(output_dir,n_i)) and iffullroll) or ('筋斗动作识别' not in os.listdir(os.path.join(output_dir,n_i)) and iffullpitch) or ('转圈动作识别' not in os.listdir(os.path.join(output_dir,n_i)) and iffullyaw) or ('攀升俯冲动作识别' not in os.listdir(os.path.join(output_dir,n_i)) and ifrapidH):
            name_ndo.append(n_i)

    if cnt_num<=len(name_ndo) and cnt_num>0:
        name_ndo=name_ndo[:cnt_num]

    nm_counter=0
    sum_plane=0
    start_time_glb = time.time()

    if prcss_max_num==0 :
        prcss_max_num=os.cpu_count()
    elif prcss_max_num==-1 :
        prcss_max_num=round(os.cpu_count()/2)
    
    while (nm_counter < len(name_ndo) and len(name_ndo)):

        ifnotexist=[1,1,1,1]
        
        source_dir=os.path.join(base_dir,name_ndo[nm_counter],'飞行数据.txt')
        name_dir=os.path.join(base_dir,name_ndo[nm_counter],'飞机编号.txt')

        start_time = time.time()
        with open(name_dir, 'r') as file:
            lines = file.readlines()
        ID_all = [line.strip() for line in lines]
        ID_all = list(dict.fromkeys(ID_all))

        try:
            os.mkdir(os.path.join(output_dir,name_ndo[nm_counter]))
        except FileExistsError :
            nothing_fee=0

        sum_plane+=len(ID_all)

        if iffullroll:
            try:
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'横滚动作识别'))
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'横滚动作识别','photo'))
            except FileExistsError :
                ifnotexist[0]=0
        if iffullpitch:
            try:
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'筋斗动作识别'))
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'筋斗动作识别','photo'))
            except FileExistsError :
                ifnotexist[1]=0
        if iffullyaw:
            try:
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'转圈动作识别'))
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'转圈动作识别','photo'))
            except FileExistsError :
                ifnotexist[2]=0
        if ifrapidH:
            try:
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'攀升俯冲动作识别'))
                os.mkdir(os.path.join(output_dir,name_ndo[nm_counter],'攀升俯冲动作识别','photo'))
            except FileExistsError :
                ifnotexist[3]=0
        
        gl_counter=0
        while(gl_counter<(len(ID_all)) and sum(ifnotexist)):

            if iffullpitch+iffullroll+iffullyaw+ifrapidH==0:
                break

            prcss_list,gl_counter=get_lotid(ID_all,prcss_max_num,gl_counter)

            if iffullroll and ifnotexist[0]:
                processes=[]
                for i in range(len(prcss_list)):
                    p = multiprocessing.Process(target=fullroll, args=(prcss_list[i], source_dir, os.path.join(output_dir,name_ndo[nm_counter],'横滚动作识别')))
                    p.start()
                    processes.append(p)
                for process in processes :
                    process.join()
            
            if iffullpitch and ifnotexist[1]:
                processes=[]
                for i in range(len(prcss_list)):
                    p = multiprocessing.Process(target=fullpitch, args=(prcss_list[i], source_dir, os.path.join(output_dir,name_ndo[nm_counter],'筋斗动作识别')))
                    p.start()
                    processes.append(p)
                for process in processes :
                    process.join()

            if iffullyaw and ifnotexist[2]:
                processes=[]
                for i in range(len(prcss_list)):
                    p = multiprocessing.Process(target=fullyaw, args=(prcss_list[i], source_dir, os.path.join(output_dir,name_ndo[nm_counter],'转圈动作识别')))
                    p.start()
                    processes.append(p)
                for process in processes :
                    process.join()
            
            if ifrapidH and ifnotexist[3]:
                processes=[]
                for i in range(len(prcss_list)):
                    p = multiprocessing.Process(target=identify_rapid_climbs_and_descents, args=(prcss_list[i], source_dir, os.path.join(output_dir,name_ndo[nm_counter],'攀升俯冲动作识别')))
                    p.start()
                    processes.append(p)
                for process in processes :
                    process.join()

            nd_t=round((time.time()-start_time)/gl_counter*(len(ID_all)-gl_counter))
            print('任务：识别 '+name_ndo[nm_counter]+' 日志中的飞行动作\t'+str(nm_counter+1)+'/'+str(len(name_ndo))+'(总任务量)\n'+'剩余：'+str(nd_t)+' 秒\t'+str(gl_counter)+'/'+str(len(ID_all))+'(单任务飞机数)')

        print('完成对 '+name_ndo[nm_counter]+' 日志中飞机动作的识别\n'+"进度<--"+str(round(nm_counter*100/len(name_ndo),2))+"%-->"+"\t预计还需要: "+str(round((time.time() - start_time_glb)*(len(name_ndo)/(nm_counter+1)-1)))+" 秒")
        nm_counter+=1

    print('总共耗费时长：'+str(round(time.time()-start_time_glb))+' 秒\t'+'执行任务量：'+str(len(name_ndo))+' 个'+'处理飞机个数：'+str(sum_plane)+' 架')



