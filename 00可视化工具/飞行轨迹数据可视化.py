# 时间数组time
# 纬度数组latitude
# 经度数组longitude
# 高度数组height
# 翻滚角度roll
# 俯仰角  pitch
# 偏航角  Yaw
# 修改日期：2024/8/24/8/24/00
import math
# import plotly.graph_objs as go
import numpy as np
import time
from tkinter import filedialog
from datetime import datetime
import os
import gc
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
output_dir=os.path.join(output_dir,'飞行轨迹'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
print('已选择'+output_dir+"作为输出地址")

def painter (ID, save_base_dir = output_dir) :

    longitude = []
    latitude = []
    height = []
    time = []
    roll = []
    pitch = []
    yaw = []

    import plotly.graph_objs as go
    # 使用方法：
    time, longitude, latitude, height, roll, pitch, yaw = requiredata(ID,source_dir)
    # 过滤掉None值
    longitude = [lng for lng in longitude if lng is not None]
    latitude = [lat for lat in latitude if lat is not None]
    height = [hght for hght in height if hght is not None]
    num=len(longitude)
    print(ID+"飞机  共有数据量："+ str(len(longitude)))

    '''
    start=0
    end=len(time)
    

    if end > len(longitude) :
        end=len(longitude)-1

    latitude= latitude[start:end]
    longitude=longitude[start:end]
    height=height[start:end]
    '''
    # 确保所有列表长度相同
    min_length = min(len(lst) for lst in [longitude, latitude, height])
    longitude = longitude[:min_length]
    latitude = latitude[:min_length]
    height = height[:min_length]

    earth_radius = 6371  # 地球的平均半径，单位：千米
    x = earth_radius * np.cos(np.radians(longitude)) * np.cos(np.radians(latitude))
    y = earth_radius * np.sin(np.radians(longitude)) * np.cos(np.radians(latitude))
    z = np.array(height)

    # 创建一个3D散点图
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
        line=dict(width=2),  # 设置线条宽度为4
        #marker=dict(size=12)
        #mode='lines+markers',  # 同时显示线和点
        name='Trajectory'
    )

    # 创建图形对象并添加trace
    fig = go.Figure(data=[trace])

    # 设置布局
    fig.update_layout(
        title='3D Trajectory',
        scene=dict(
            xaxis=dict(title='X (km)'),
            yaxis=dict(title='Y (km)'),
            zaxis=dict(title='Altitude (km)')
        )
    )

    # 显示图形
    #fig.show()

    if num <=500 :
        save_base_dir=os.path.join(save_base_dir,'0_to_500')
    elif num<=1000 :
        save_base_dir=os.path.join(save_base_dir,'500_to_1000')
    elif num<=2000 :
        save_base_dir=os.path.join(save_base_dir,'1000_to_2000')
    elif num<=3000 :
        save_base_dir=os.path.join(save_base_dir,'2000_to_3000')
    elif num<=5000 :
        save_base_dir=os.path.join(save_base_dir,'3000_to_5000')
    else:
        save_base_dir=os.path.join(save_base_dir,'5000_to_infinity')

    fig.write_html(os.path.join(save_base_dir,ID+'_('+str(num)+").html"))

    del trace
    del fig
    del go
    del time
    del longitude
    del latitude
    del height
    del roll
    del pitch
    del yaw
    gc.collect()

with open(name_dir, 'r') as file:
    lines = file.readlines()
ID_all = [line.strip() for line in lines]
start_time = time.time() 
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir,'0_to_500'))
os.mkdir(os.path.join(output_dir,'500_to_1000'))
os.mkdir(os.path.join(output_dir,'1000_to_2000'))
os.mkdir(os.path.join(output_dir,'2000_to_3000'))
os.mkdir(os.path.join(output_dir,'3000_to_5000'))
os.mkdir(os.path.join(output_dir,'5000_to_infinity'))

counter=0
for id in ID_all :
    painter(id)
    counter=counter+1
    end_time = time.time()
    print("已完成 "+id+" 号飞机的图象绘制。"+"\n进度<--"+str(round(counter*100/len(ID_all),2))+"%-->"+"\t预计还需要: "+str(round((end_time - start_time)/(counter/len(ID_all))*(1-counter/len(ID_all))))+" 秒")

end_time = time.time()
print("程序运行时间: "+str(round(end_time - start_time))+" 秒")