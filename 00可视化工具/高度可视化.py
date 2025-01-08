# 时间数组time
# 纬度数组latitude
# 经度数组longitude
# 高度数组height
# 翻滚角度roll
# 俯仰角  pitch
# 偏航角  Yaw
# 修改日期：2024/8/24/8/24/00
from cllct_data_of_a_partclr_pln import requiredata
    
source_dir='C:\\000 onedrive 备份\\OneDrive\\桌面\\空战建模答辩\\飞行数据 - 副本.txt'

# 使用方法：
time, longitude, latitude, height, roll, pitch, yaw = requiredata('305',source_dir)

def printphoto(height,time):

    import matplotlib.pyplot as plt
    plt.close()
    start=50
    end=2000
    # 假设 height 是一个已经存在的列表，包含了高度数据
    height=height[start:end]
    # 生成与 height 列表长度相同的下标序列
    time=time[start:end]

    plt.figure(figsize=(15, 8))
    plt.scatter(time, height, color='blue', marker='o')
    plt.title('Change of altitude over time')
    plt.xlabel('time')
    plt.ylabel('height')
    plt.grid(True)
    plt.show()

    return None

printphoto (roll,time)