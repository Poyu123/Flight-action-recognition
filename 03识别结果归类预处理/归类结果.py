import os 
import shutil
import time
from tqdm import tqdm
from tkinter import filedialog

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

name_01=os.listdir(base_dir)

min=0
max=1
for name_01_t in name_01 :
    tempt=int(name_01_t[0]+name_01_t[1]+name_01_t[2])
    if tempt<min :
        min=tempt
    if tempt>max :
        max=tempt

output_dir=os.path.join(output_dir,'动作识别汇总('+str(min)+'_'+str(max)+')')

counter=0

for name_01_t in name_01 :
     name_02=os.listdir(os.path.join(base_dir,name_01_t))
     for name_02_t in name_02 :
         name_03=[i for i in os.listdir(os.path.join(base_dir,name_01_t,name_02_t)) if i != 'photo' ]
         for name_03_t in name_03 :
             name_04=os.listdir(os.path.join(base_dir,name_01_t,name_02_t,name_03_t))
             for name_04_t in name_04 :
                 counter+=1

print("进行中...")
progress_bar = tqdm(total=counter)
start_time=time.time()

for name_01_t in name_01 :
     name_02=os.listdir(os.path.join(base_dir,name_01_t))
     for name_02_t in name_02 :
         name_03=[i for i in os.listdir(os.path.join(base_dir,name_01_t,name_02_t)) if i != 'photo' ]
         for name_03_t in name_03 :
             name_04=os.listdir(os.path.join(base_dir,name_01_t,name_02_t,name_03_t))
             for name_04_t in name_04 :
                 destination_path=os.path.join(output_dir,name_02_t)
                 if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                 shutil.copy(os.path.join(base_dir,name_01_t,name_02_t,name_03_t,name_04_t),os.path.join(destination_path,name_01_t+'_'+name_03_t+'_'+name_04_t))
                 progress_bar.update(1)
             
progress_bar.close()
print('已完成。\n共耗费时长: '+str(round(time.time()-start_time)))