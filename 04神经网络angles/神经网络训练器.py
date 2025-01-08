from keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

base_dir=''
data_path='C:\\Users\\Poter\\Desktop\\训练数据\\data(C=200_F=25_L=50_A=4).npz'#单次运行地址

def MAIN(DATA_PATH):

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 32
    epochs_num=25


    NUM_C=0
    NUM_F=0
    NUM_A=0
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
        elif DATA_PATH[i] == 'F' and DATA_PATH[i+1] == '=':
            counter_p=0
            while(1):
                if DATA_PATH[i+2+counter_p].isdigit():
                    NUM_F=int(DATA_PATH[i+2+counter_p])+NUM_F*10
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
    
    #读取数据
    with np.load(DATA_PATH) as data:
        train_examples = data['dtrn']
        train_labels = data['lbltrn']

    #分割一部分数据作为测试
    split_index=round(len(train_examples)/10)
    test_examples = train_examples[:split_index]
    test_labels = train_labels[:split_index]
    train_examples = train_examples[split_index:]
    train_labels = train_labels[split_index:]

    '''
    print(test_examples.shape)
    print(test_examples[0].shape)
    print(test_examples[[0]].shape)
    '''

    #配置数据生成器
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    #配置层数神经元个数
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(NUM_F,6)),
        tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(NUM_A, activation = 'softmax')
    ])

    #配置编译
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    #训练
    model.summary()
    history=model.fit(train_dataset, epochs=epochs_num)

    print('测试准确度')
    x=model.evaluate(test_dataset)

    return history,x

def painter(history,X):

        acc = history.history['sparse_categorical_accuracy']
        los = history.history['loss']
        epochs_num=len(acc)

        epochs_range = range(epochs_num)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.scatter(epochs_num, X[1], color='red', label='Validation Accuracy') 
        plt.text(epochs_num, X[1]-0.01, str(round(X[1],3)), ha='center', va='bottom') 
        plt.legend(loc='lower right')
        plt.title('Training Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, los, label='Training Loss')
        plt.scatter(epochs_num, X[0], color='red', label='Validation loss') 
        plt.text(epochs_num, round((X[0]-0.5),2), str(round(X[0],3)), ha='center', va='bottom') 
        plt.legend(loc='upper right')
        plt.title('Training Loss')
        #plt.savefig(os.path.join(model_dir,'acc='+str(round(acc[epochs_num-1]*100)/100)+'  val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+'.png'))
        plt.show()



ifmass=0#是否集中训练

if ifmass:
    name01=os.listdir(base_dir)
    for name_01_t in name01:
        data_path=os.path.join(base_dir,name_01_t)
        MAIN(data_path)
else:
    history, X=MAIN(data_path)
    painter(history, X)