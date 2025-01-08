# 修改日期：2024/8/24/22/00/00

import os
import zipfile
import shutil

start=0

def number(start,counter):
    num=start+counter
    if num<10 :
        strnum='00'+str(num)
    elif num<100 :
        strnum='0'+str(num)
    else:
        strnum=str(num)
    return strnum

download_dir="C:\\000 onedrive 备份\\OneDrive\\桌面\\飞机数据\\下载原始数据"
output_dir="C:\\000 onedrive 备份\\OneDrive\\桌面\\飞机数据\\原始数据"
start=len(os.listdir(output_dir))+1
old_path=os.listdir(download_dir)
new_path=[i.split('.acmi')[0] for i in old_path]
for counter in range(len(old_path)):
    os.rename(os.path.join(download_dir,old_path[counter]), os.path.join(download_dir,new_path[counter]))

old_path=[os.path.join(download_dir,i) for i in new_path]

new_path=[os.path.join(download_dir,i[7:15]) for i in new_path]

counter=0
for i in range(len(old_path)-1):
    with zipfile.ZipFile(old_path[counter], 'r') as zip_ref:
        zip_ref.extractall(os.path.join(download_dir,'tempt'))
        shutil.move(os.path.join(download_dir,'tempt',os.listdir(os.path.join(download_dir,'tempt'))[0]),os.path.join(output_dir,number(start,counter)+'_Tacview.acmi'))
    os.remove(old_path[counter])
    counter+=1

print("完成")