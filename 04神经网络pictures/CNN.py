import os
import time
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models, regularizers

start_time=time.time()

tf.debugging.set_log_device_placement(True)

def getvar (DATA_PATH):
    NUM_C=0
    NUM_L=0
    NUM_A=0
    NUM_X=0
    NUM_Y=0
    NUM_P=0
    counter_p=0
    for i in range(len(DATA_PATH)):
        if DATA_PATH[i] == 'C' and DATA_PATH[i+1] == '=':
            counter_p=0
            while(1):
                if DATA_PATH[i+2+counter_p].isdigit():
                    NUM_C=int(DATA_PATH[i+2+counter_p])+NUM_C*10
                    counter_p+=1
                else:
                    break
        elif DATA_PATH[i] == 'L' and DATA_PATH[i+1] == '=':
            counter_p=0
            while(1):
                if DATA_PATH[i+2+counter_p].isdigit():
                    NUM_L=int(DATA_PATH[i+2+counter_p])+NUM_L*10
                    counter_p+=1
                else:
                    break
        elif DATA_PATH[i] == 'A' and DATA_PATH[i+1] == '=':
            counter_p=0
            while(1):
                if DATA_PATH[i+2+counter_p].isdigit():
                    NUM_A=int(DATA_PATH[i+2+counter_p])+NUM_A*10
                    counter_p+=1
                else:
                    break
        elif DATA_PATH[i] == 'X' and DATA_PATH[i+1] == '=':
            counter_p=0
            while(1):
                if DATA_PATH[i+2+counter_p].isdigit():
                    NUM_X=int(DATA_PATH[i+2+counter_p])+NUM_X*10
                    counter_p+=1
                else:
                    break
        elif DATA_PATH[i] == 'Y' and DATA_PATH[i+1] == '=':
            counter_p=0
            while(1):
                if DATA_PATH[i+2+counter_p].isdigit():
                    NUM_Y=int(DATA_PATH[i+2+counter_p])+NUM_Y*10
                    counter_p+=1
                else:
                    break
        elif DATA_PATH[i] == 'P' and DATA_PATH[i+1] == '=':
            counter_p=0
            while(1):
                if DATA_PATH[i+2+counter_p].isdigit():
                    NUM_P=int(DATA_PATH[i+2+counter_p])+NUM_P*10
                    counter_p+=1
                else:
                    break

    return NUM_C,NUM_L,NUM_A,NUM_X,NUM_Y,NUM_P

epochs_num=500
batch_size_num=256
patience_num=24

base_dir='C:\\000 onedrive 备份\\OneDrive\\桌面\\3figures(C=1000_L=50_A=4_X=64_Y=64_P=8_T=20_O=750)'

NUM_C,NUM_L,NUM_A,NUM_X,NUM_Y,NUM_P=getvar(base_dir)

early_stopping =tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_num, verbose=1)

train_dir=os.path.join(base_dir,'train')
test_dir=os.path.join(base_dir,'test')

train_dategen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.00)
test_dategen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.00)

train_generator=train_dategen.flow_from_directory(
    train_dir,
    target_size=(NUM_X,NUM_Y),
    batch_size=batch_size_num,
    class_mode='categorical'
)

test_generator=test_dategen.flow_from_directory(
    test_dir,
    target_size=(NUM_X,NUM_Y),
    batch_size=batch_size_num,
    class_mode='categorical'
)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',padding='same', input_shape = (NUM_X,NUM_Y,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(256, (3,3), activation = 'relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(2964, activation = 'relu',kernel_regularizer=regularizers.l2(0.005)))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(296, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(NUM_A, activation = 'softmax'))
    
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()

if (os.path.exists(os.path.join(base_dir,'model'))) == 0:
    os.mkdir(os.path.join(base_dir,'model'))

history = model.fit(
                    train_generator,
                    epochs=epochs_num, 
                    validation_data=test_generator,
                    callbacks=[early_stopping],
                    verbose=1
                    )

#print(history.history)

val_acc = history.history['val_categorical_accuracy']
epochs_num=val_acc.index(max(val_acc))+1
val_acc=val_acc[:epochs_num]
acc = history.history['categorical_accuracy'][:epochs_num]
loss = history.history['loss'][:epochs_num]
val_loss = history.history['val_loss'][:epochs_num]


nowtime = time.strftime('%Y_%m_%d_%H_%M_%S')

model_dir=os.path.join(base_dir,'model',str(nowtime)+'(val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+')')
os.mkdir(model_dir)

with open(os.path.join(model_dir,'model.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

epochs_range = range(epochs_num)

# 绘制准确率图表
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.scatter(epochs_range[-1], acc[-1], color='blue')
plt.text(epochs_range[-1], acc[-1], f'{acc[-1]:.2f}', color='blue', ha='right', va='bottom', fontweight='bold')
plt.scatter(epochs_range[-1], val_acc[-1], color='orange')
plt.text(epochs_range[-1], val_acc[-1], f'{val_acc[-1]:.2f}', color='orange', ha='right', va='bottom', fontweight='bold')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# 绘制损失图表
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.scatter(epochs_range[-1], loss[-1], color='blue')
plt.text(epochs_range[-1], loss[-1], f'{loss[-1]:.2f}', color='blue', ha='right', va='top', fontweight='bold')
plt.scatter(epochs_range[-1], val_loss[-1], color='orange')
plt.text(epochs_range[-1], val_loss[-1], f'{val_loss[-1]:.2f}', color='orange', ha='right', va='top', fontweight='bold')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig(os.path.join(model_dir,'acc='+str(round(acc[epochs_num-1]*100)/100)+'  val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+'.png'))
print("\n\n\nThe acc="+str(round(acc[epochs_num-1]*100)/100)+'\nval_acc='+str(round(val_acc[epochs_num-1]*100)/100)+'\nepoch='+str(epochs_num))
print ("consuming minutes are :"+str(round((time.time()-start_time)/60,2))+'\tmin')

if (round(acc[epochs_num-1]*100)/100) >= 95 :
    confirm=input("save?[y/n](recommended):")
else:
    confirm=input("save?[y/n]:")
if confirm=='y':
    model.save(os.path.join(model_dir,'model'+'(X='+str(NUM_X)+'_Y='+str(NUM_Y)+').h5'))
else:
    confirm=input("delete?[y/n]:")
    if confirm=='y':
        shutil.rmtree(model_dir)