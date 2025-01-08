# 修改日期：2024/8/24/8/24/00

import os
from tqdm import tqdm
from tkinter import filedialog

print("请选择数据来源：")
source_dir=''
while not source_dir :
    source_dir=filedialog.askdirectory()
print('已选择'+source_dir+"作为数据来源")
print("请选择输出地址：")
output_dir=''
while not output_dir :
    output_dir = filedialog.askdirectory()
print('已选择'+output_dir+"作为输出地址")
opbase_dir=output_dir
source_name=[i.split('.acmi')[0] for i in os.listdir(source_dir)]
exsist_name=os.listdir(output_dir)
counter=0
for name_tempt in source_name :
    if name_tempt not in exsist_name :
        counter+=1
numall=counter
counter=0

def identity(line):
    parts = line.split(',')  # 将原始数据用逗号分割提取ID（飞机的）
    str_identity = 'Type=Air+FixedWing'
    if len(parts) > 2 and parts[2] == str_identity:
        return True
    else:
        return False
    
for name_tempt1 in source_name :
    name_tempt=name_tempt1
    if name_tempt not in exsist_name :
        counter+=1
        output_dir=os.path.join(opbase_dir,name_tempt)
        name_tempt=os.path.join(source_dir,name_tempt+'.acmi')
        mode=0#是否采用“UTF-8”解码

        try:
            data = open( name_tempt, "rt")
            # 查找是飞机的id
            airplane_id = []
            for line in data.readlines():
                if identity(line):
                    parts = line.split(',')
                    airplane_id.append(parts[0])  # 将飞机的id全部记录下来
            data.close()
            mode=0
        except UnicodeDecodeError :
            data = open( name_tempt, "rt",encoding='utf-8')
            # 查找是飞机的id
            airplane_id = []
            for line in data.readlines():
                if identity(line):
                    parts = line.split(',')
                    airplane_id.append(parts[0])  # 将飞机的id全部记录下来
            data.close()
            mode=1

        airplane_id=list(dict.fromkeys(airplane_id))
        print('一共有'+str(len(airplane_id))+'架飞机。正在加速读取中...\t'+str(counter)+' / '+str(numall))
        progress_bar = tqdm(total=len(airplane_id),position=0)
        os.mkdir(output_dir)
        with open(os.path.join(output_dir,'飞行数据.txt'), 'w') as save_file:

            for ID in airplane_id:
                save_file.write('\n#the airplane is named'+' '+ID+'\n')#告诉记录的飞机是谁
                if mode :
                    data = open( name_tempt, "rt",encoding='utf-8')
                else:
                    data = open( name_tempt, "rt")
                for line in data.readlines():
                    if '#'in line:
                        time=line.replace('#','@time=')#记录时间，特殊符号@来间隔
                    else:
                        parts = line.split(',')
                        if ID == parts[0] and len(parts)<3:
                            save_file.write(line.rstrip('\n')+time)#提取是这个飞机的行数
                data.close()
                progress_bar.update(1)
                #print('完成'+ID+"号飞机的数据收集"+'（第'+str(airplane_id.index(ID)+1)+'架战斗机）')

        with open(os.path.join(output_dir,'飞机编号.txt'), 'wt') as save_file:
            for ID in airplane_id:
                save_file.write(ID+'\n')
        progress_bar.close() 

print('运行完成')
