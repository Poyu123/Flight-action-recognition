import os
import time
import shutil
from tqdm import tqdm

def getdata (name) :
    num_folder=''
    name_plane=''
    tokens=name.split('_')
    num_folder=tokens[0]
    name_plane=tokens[2]
    mode=0
    if tokens[3]=='log.txt':
        mode=1
    return num_folder, name_plane, mode

source_dir = "C:\\Users\\Poter\\Desktop\\动作识别汇总(0_2)\\攀升俯冲动作识别"
output_dir = 'C:\\Users\\Poter\\Desktop'

start_t=time.time()
print('进行中...')
os.mkdir(os.path.join(output_dir,'攀升动作识别'))
os.mkdir(os.path.join(output_dir,'俯冲动作识别'))
counter=0
name01=os.listdir(source_dir)
amount=len(name01)
for name01_t in name01 :
    num_folder, name_plane, mode=getdata(name01_t)
    if mode :
        amount-=1
progress_bar = tqdm(total=amount)

for name01_t in name01 :
    num_folder, name_plane, mode=getdata(name01_t)
    if mode :
        with open(os.path.join(source_dir,name01_t),'r') as file:
            for line_t in file.readlines():
                line=line_t.replace('\n', '')
                if line[:4] == '快速上升':
                    #print('1')
                    action='攀升动作识别'
                elif line[:4] == '快速下降':
                    #print('2')
                    action='俯冲动作识别'
                elif line [:4]=='开始时间' :
                    #print('3')
                    start_time=line[5:]
                elif line [:4]=='结束时间' :
                    #print('4')
                    end_time=line[5:]
                elif line [:4]=='平均爬升' or line [:4]=='平均下降':
                    #print('5')
                    old_name=num_folder+'_Tacview_'+name_plane+'_'+start_time+'_'+'to'+'_'+end_time+'.txt'
                    #print(old_name)
                    if os.path.exists(os.path.join(source_dir,old_name)):
                        shutil.move(os.path.join(source_dir,old_name),os.path.join(output_dir,action,old_name))
                    else:
                        counter+=1
                    progress_bar.update(1)

progress_bar.close()
print('已完成。'+' '+'总共耗费时长：'+str(round(time.time()-start_t))+' 秒。')
print('缺失数据：'+str(counter))

