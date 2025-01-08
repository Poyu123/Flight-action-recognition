# 修改日期：2024/8/24/22/00/00

import os
import time
import shutil
import multiprocessing
from tkinter import filedialog

def identity(line):
    parts = line.split(',')  # 将原始数据用逗号分割提取ID（飞机的）
    str_identity = 'Type=Air+FixedWing'
    if len(parts) > 2 and parts[2] == str_identity:
        return True
    else:
        return False

def get_name(ID_all,prcss_max_num,gl_counter):
    if len(ID_all[gl_counter:])>=prcss_max_num :
        prcss_idlist=[ID_all[gl_counter+i] for i in range(prcss_max_num)]
        gl_counter=gl_counter+prcss_max_num
    else :
        prcss_idlist=[ID_all[gl_counter+i] for i in range(len(ID_all[gl_counter:]))]
        gl_counter=gl_counter+len(prcss_idlist)
    return prcss_idlist, gl_counter

def writedata(name_tempt,opbase_dir,source_dir):
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

        os.mkdir(output_dir)

        with open(os.path.join(output_dir,'未完成.txt'), 'wt') as save_file:
            save_file.write('未完成')

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

        with open(os.path.join(output_dir,'飞机编号.txt'), 'wt') as save_file:
            for ID in airplane_id:
                save_file.write(ID+'\n')

        os.remove(os.path.join(output_dir,'未完成.txt'))

if __name__ == '__main__':   

    prcss_max_num=12#进程数量
    ifclean=1#是否删除未完成的
    start_num=0
    end_num=82

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
    
    if ifclean:
        for name_e in exsist_name:
            if '未完成.txt' in os.listdir(os.path.join(output_dir,name_e)):
                shutil.rmtree(os.path.join(output_dir,name_e))
        exsist_name=os.listdir(output_dir)

    counter=0
    start_time=time.time()
    source_name_c=source_name
    source_name=[]
    for name_tempt in source_name_c :
        if name_tempt not in exsist_name and int(name_tempt[0]+name_tempt[1]+name_tempt[2])<=end_num and start_num<=int(name_tempt[0]+name_tempt[1]+name_tempt[2]):
            source_name.append(name_tempt)
            counter+=1
    source_name_c=[]
    numall=counter
    counter=0
    gn_counter=0

    while(gn_counter<(len(source_name))):
        prcss_list,gn_counter=get_name(source_name,prcss_max_num,gn_counter)
        processes=[]
        for i in range(len(prcss_list)):
            p = multiprocessing.Process(target=writedata, args=(prcss_list[i], opbase_dir, source_dir))
            p.start()
            processes.append(p)
        for process in processes :
            process.join()

        nd_t=round((time.time()-start_time)/gn_counter*(len(source_name)-gn_counter))
        print("进度<--"+str(round(gn_counter*100/len(source_name),2))+"%-->"+"\t预计还需要: "+str(nd_t)+" 秒")

    print('总共用时：'+str(round(time.time()-start_time))+' 秒。\t转换'+str(gn_counter)+'个文件')

