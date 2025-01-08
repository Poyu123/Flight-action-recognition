#间数组time_1
# 纬度数组latitude
# 经度数组longitude
# 高度数组height
# 翻滚角度roll
# 俯仰角  pitch
# 偏航角  Yaw
# 修改日期：2024/8/24/8/24/00

import time
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog
from cllct_data_of_a_partclr_pln import requiredata

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
output_dir=os.path.join(output_dir,os.path.basename(os.path.dirname(source_dir))[:3]+'_筋斗识别'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
print('已选择'+output_dir+"作为输出地址")

"""
source_dir = 'C:\\000 onedrive 备份\\OneDrive\桌面\飞机数据\\001_Tacview\\飞行数据.txt'
name_dir = 'C:\\000 onedrive 备份\\OneDrive\桌面\飞机数据\\001_Tacview\\飞机编号.txt'
output_dir = 'C:\\000 onedrive 备份\\OneDrive\\桌面'
output_dir = os.path.join(output_dir, '攀升俯冲识别' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
"""

def calculate_average_pitch(pitch, start_index, end_index):
    total_pitch = sum(pitch[start_index:end_index + 1])
    average_pitch = total_pitch / (end_index - start_index + 1)
    return average_pitch


import os


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

start_time = time.time()

with open(name_dir, 'r') as file:
    lines = file.readlines()
ID_all = [line.strip() for line in lines]
ID_all = list(dict.fromkeys(ID_all))
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, 'photo'))
counter = 0
for id in ID_all:
    identify_rapid_climbs_and_descents(id, source_dir, output_dir)
    counter = counter + 1
    end_time = time.time()
    print("进度<--" + str(round(counter * 100 / len(ID_all), 2)) + "%-->" + "\t预计还需要: " + str(
        round((end_time - start_time) / (counter / len(ID_all)) * (1 - counter / len(ID_all)))) + " 秒")

end_time = time.time()
print("程序运行时间: " + str(round(end_time - start_time)) + " 秒")

