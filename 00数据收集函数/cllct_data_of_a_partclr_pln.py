#时间数组time_1
#纬度数组latitude
#经度数组longitude
#高度数组height
#翻滚角度roll
#俯仰角  pitch
#偏航角  Yaw
# 修改日期：2024/8/24/8/24/00

def requiredata(ID,source_dir):  # ID是文本！！

    data = open(source_dir, "rt")
    on_off = 0  # 开关用来记录是否开始
    counter=0

    for line in data.readlines():
        if ('#the airplane is named ' + ID) in line:  # 注意ID是文本
            on_off = 1
        elif on_off == 1 and line[0] == '#':  # 如果识别到了目标飞机的下一个飞机就跳出来
            break
        elif on_off == 1 and ('@' in line):  # 开始提取数据
            counter+=1

    data.close()
    data = open(source_dir, "rt")
    on_off=0

    counter+=1
    longitude = [0.0]*counter
    latitude = [0.0]*counter
    height = [0.0]*counter
    time_1 = [0.0]*counter
    roll = [0.0]*counter
    pitch = [0.0]*counter
    yaw = [0.0]*counter
    counter=0

    for line in data.readlines():
        if ('#the airplane is named ' + ID) in line:  # 注意ID是文本
            #print('开始收集数据')
            on_off = 1
        elif on_off == 1 and line[0] == '#':  # 如果识别到了目标飞机的下一个飞机就跳出来
            break
        elif on_off == 1 and ('@' in line):  # 开始提取数据
            line=line.strip()
            counter+=1
            parts_1 = line.split('@')  # 先分出时间
            parts_2 = parts_1[0].split(',')  # 分出标号
            parts = parts_2[1].lstrip('T=')  # 去掉‘T=’符号
            #print(parts)
            #time.sleep(1)
            basedata = parts.split('|')  # 分割几种（经纬高...）数据

            try:
                lose_time = 0
                time_1_temp = float(parts_1[1].lstrip('time_1='))  # 得出浮点数的时间
                time_1[counter]=(time_1_temp)  # 时间存入数组
            except ValueError:
                if time_1[1]!=0:
                    time_1[counter]=(time_1[counter-1])
                else:
                    lose_time = 1  # 上一个数据丢失
                    time_1[counter]=(None)
            except IndexError:
                if time_1[1]!=0:
                    time_1[counter]=(time_1[counter-1])
                else:
                    lose_time = 1  # 上一个数据丢失
                    time_1[counter]=(None)

            try:
                lose_lngi = 0
                longitude[counter]=(float(basedata[0]))  # 存储经度数据
            except ValueError:
                if longitude[1]!=0:
                    longitude[counter]=(longitude[counter-1])
                else:
                    lose_lngi = 1  # 上一个数据丢失
                    longitude[counter]=(None)
            except IndexError:
                if longitude[1]!=0:
                    longitude[counter]=(longitude[counter-1])
                else:
                    lose_lngi = 1  # 上一个数据丢失
                    longitude[counter]=(None)

            try:
                lose_lati = 0
                latitude[counter]=(float(basedata[1]))  # 存储纬度数据
            except ValueError:
                if latitude[1]!=0:
                    latitude[counter]=(latitude[counter-1])
                else:
                    lose_lati = 1  # 上一个数据丢失
                    latitude[counter]=(None)
            except IndexError:
                if latitude[1]!=0:
                    latitude[counter]=(latitude[counter-1])
                else:
                    lose_lati = 1  # 上一个数据丢失
                    latitude[counter]=(None)

            try:
                lose_hght = 0
                height[counter]=(float(basedata[2]))  # 存储高度数据
            except ValueError:
                if height[1]!=0:
                    height[counter]=(height[counter-1])
                else:
                    lose_hght = 1  # 上一个数据丢失
                    height[counter]=(None)
            except IndexError:
                if height[1]!=0:
                    height[counter]=(height[counter-1])
                else:
                    lose_hght = 1  # 上一个数据丢失
                    height[counter]=(None)

            try:
                lose_roll = 0
                #print(str(basedata[3])+' @@ '+str(float(basedata[3])))
                roll[counter]=(float(basedata[3]))  # 存储翻滚角数据
            except ValueError:
                if roll[1]!=0:
                    roll[counter]=(roll[counter-1])
                else:
                    lose_roll = 1  # 上一个数据丢失
                    roll[counter]=(None)
            except IndexError:
                if roll[1]!=0:
                    roll[counter]=(roll[counter-1])
                else:
                    lose_roll = 1  # 上一个数据丢失
                    roll[counter]=(None)

            try:
                lose_pitch = 0
                pitch[counter]=(float(basedata[4]))  # 存储俯仰角数据
            except ValueError:
                if pitch[1]!=0:
                    pitch[counter]=(pitch[counter-1])
                else:
                    lose_pitch = 1  # 上一个数据丢失
                    pitch[counter]=(None)
            except IndexError:
                if pitch[1]!=0:
                    pitch[counter]=(pitch[counter-1])
                else:
                    lose_pitch = 1  # 上一个数据丢失
                    pitch[counter]=(None)

            try:
                lose_yaw = 0
                yaw[counter]=(float(basedata[5]))  # 存储偏航角数据
            except ValueError:
                if yaw[1]!=0:
                    yaw[counter]=(yaw[counter-1])
                else:
                    lose_yaw = 1  # 上一个数据丢失
                    yaw[counter]=(None)
            except IndexError:
                if yaw[1]!=0:
                    yaw[counter]=(yaw[counter-1])
                else:
                    lose_yaw = 1  # 上一个数据丢失
                    yaw[counter]=(None)

            if (lose_lngi + lose_lati + lose_hght + lose_roll + lose_pitch + lose_yaw + lose_time):
                del time_1[counter]
                del longitude[counter]
                del latitude[counter]
                del height[counter]
                del roll[counter]
                del pitch[counter]
                del yaw[counter]
                counter-=1
                #print("删除！！2")
            elif (abs(roll[counter])>180 or abs(pitch[counter])>180 or yaw[counter]>360):
                del time_1[counter]
                del longitude[counter]
                del latitude[counter]
                del height[counter]
                del roll[counter]
                del pitch[counter]
                del yaw[counter]
                counter-=1

            #print(roll[counter])
                
    data.close()

    counter=0
    num=len(longitude)
    while(counter<num):
        if longitude[counter]==0 and latitude[counter]==0 and height[counter]==0 and roll[counter]==0 and pitch[counter]==0 and yaw[counter]==0 :
            counter+=1
        else:
            break
    if counter==num :
        counter-=1
    
    time_1=time_1[counter:]
    longitude=longitude[counter:]
    latitude=latitude[counter:]
    height=height[counter:]
    roll=roll[counter:]
    pitch=pitch[counter:]
    yaw=yaw[counter:]

    return time_1, longitude, latitude, height, roll, pitch, yaw

'''
#使用方法：
time_1 ,longitude ,latitude ,height ,roll ,pitch ,yaw= requiredata('602',source_dir)
print(pitch)
'''