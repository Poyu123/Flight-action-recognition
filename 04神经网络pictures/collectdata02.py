#时间数组time_1
#纬度数组latitude
#经度数组longitude
#高度数组height
#翻滚角度roll
#俯仰角  pitch
#偏航角  Yaw
# 修改日期：2024/8/26/22/00/00



def RQDATA (source_dir):

    longitude = []
    latitude = []
    height = []
    time = []
    roll = []
    pitch = []
    yaw = []

    data = open(source_dir, "rt")

    for line in data.readlines():
        parts=line.split(' | ')
        longitude.append(parts[0])
        latitude.append(parts[1])
        height.append(parts[2])
        roll.append(parts[3])
        pitch.append(parts[4])
        yaw.append(parts[5])
        time.append(parts[6])

    time=[float(i) for i in time]
    longitude=[float(i) for i in longitude]
    latitude=[float(i) for i in latitude]
    height=[float(i) for i in height]
    roll=[float(i) for i in roll]
    pitch=[float(i) for i in pitch]
    yaw=[float(i) for i in yaw]
    
    return longitude ,latitude ,height ,roll ,pitch ,yaw ,time

'''
source_dir="C:\\000 onedrive 备份\\OneDrive\\桌面\\测试文件夹\\001_Tacview\\横滚动作识别\\1ec012\\4338.1_to_4350.51.txt"
longitude ,latitude ,height ,roll ,pitch ,yaw ,time_1=RQDATA(source_dir)    
print(longitude)
'''
