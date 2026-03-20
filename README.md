# Flight-action-recognition
This proj. introduces two db construction methods and recognition methods, achieving fast and accurate aircraft movement recognition. It starts with flight segment machine ID, then integrates labelled aircraft fragments for training and verification db construction, and trains FCNN and CNN. The method exhibits excellent recognition capabilities.

# Preface
This work is really interesting but a bit challenging. Anyway we did a lot of things and these potentially give you a hand. First of all, I would like to declare that all of data we used are from a nonearning cloud belonged to Glowlingsever. Here is its website address : https://drive.google.com/drive/folders/178nooqDx1yNS9VZ01ioOjfjEXe_AC1E9. And those data really play a significant role in our work since it is not necessary to fly a plane on our own! We took advantage of around 50Gb to do this research. This means that you'd better to follow this way to save time.

# Download
I will introduce the code file **'downloadpresolve.py'**, let you know how this file functions and simply talk about its purpose. And please remember that this type of method is also utilised in the left '.py' files.
First, you need to download such a 'raw' document ending with **'.zip.acmi'** through the above website mentioned in Preface. Then, change the addresses in downloadpresolve.py of two variables named **'download_dir' and 'output_dir'** into yours. Please be careful since these two addresses must ***exist*** and they are only two ***folders***.  If you succeed, there will be such a file ending only with **'.acmi'**. And this is what we truely need. This .py file achieves the action of decompressing. One more important thing is that you'd better **backup file** to lessen the adverse impacts like computer crash though there is only slight risk.  

# Organise
Please run the code file **'solvedata.py'**. Once you have done the part Download, you would get a pile of files with the end of '.acmi'. And their folder is the source which is the first folder you selected. Mention : please only select ***folder***. The next one you select is the existing folder whatever you want. For a single file, it might takes 1 to 2 mins to complete. If you think this is time consuming, you could update this and inform me. And there is a really embarrassing problem my teammate met caused by ***tqdm***. To solve this, the best way is to install **anaconda** which is totally free and of help. If your team do not encouter this, you are ,indeed, lucky. If all steps are down, you will have access to such a type of document we defined, which is the base of all this project. 
note: if your computer owes **a powerfull cpu**, please try to use 'solvedata_多进程版.py'. Set the variable prcss_max_num to the num. no more than the number of core in cpu. And set the start number, usually **0**, and the end number depenging on the number of your files. And we recommend to set ifclean to be **1**.

# Tutorial Videos
Please click [here](https://pan.baidu.com/s/1zX-o9O5NkFVzAS_DHWGj_A?pwd=xxsm) to watch the tutorials.

# Preprocessed Data
Please click [here](https://pan.baidu.com/s/1ruI0Q5uD7yv_09Fc93ktoA?pwd=xxsm) to access to the flights data that have already been processed before being set as the input into neural networks.

## Visualization Tools
In the visualization folder, we provide several scripts to help you intuitively understand the raw numerical data.

* **3D Trajectory Visualization:** The file `飞行轨迹数据可视化.py` maps the longitude, latitude, and altitude of specific aircraft into an interactive 3D scatter plot using Plotly.
* **Altitude Tracking:** The script `高度可视化.py` plots altitude changes over time using Matplotlib.
* **Aircraft 3D Modeling:** `绘制飞机模型3D图像.py` generates a basic 3D geometric representation of the aircraft to help visualize its posture dynamically.

## Rule-Based Action Recognition
Before training neural networks, we must extract specific, meaningful maneuvers from the continuous flight data. The scripts in the action recognition folder act as a probe to slice the data based on physical rules and mathematical thresholds.

* We identify specific actions by analyzing continuous monotonicity and variance limits: Roll (`横滚识别.py`), Loop/Pitch (`筋斗识别.py`), Circle/Yaw (`转圈识别.py`), and Rapid Climb/Descent (`快速上升下降识别.py`).
* **Multiprocessing:** To process large datasets efficiently, we highly recommend using the multiprocessing version (`动作识别_多进程版.py`). This script leverages multiple CPU cores to perform the action extraction simultaneously, significantly saving time.

## Result Classification and Preprocessing
Once the fragmented action data is generated, scripts like `归类结果.py` and `攀升俯冲分离.py` help aggregate and clean the results. For example, it separates generic rapid altitude changes into distinct "Climb" and "Descent" directories, preparing a clean, labeled database for the neural networks.

## Deep Learning Models
We approached the recognition task using two distinct deep learning methods.

### Approach 1: FCNN with Angle Features
This method treats the flight data as numerical time-series arrays.

* **Data Generator (`神经网络数据生成器.py`):** Converts the raw spatial coordinates into relative angle features. This makes the data translation-invariant and focuses strictly on the aircraft's relative movements.
* **Model Trainer (`神经网络训练器.py`):** Feeds these engineered angle features into a Fully Connected Neural Network (FCNN) to classify the flight maneuvers.

### Approach 2: CNN with 2D Image Projection
This is a novel approach that converts the time-series tracking problem into an image classification problem, allowing us to leverage computer vision techniques.

* **Image Generator (`投影彩色图片化生成器.py`):** Uses Principal Component Analysis (PCA) to project the 3D flight trajectory onto optimal 2D planes. It assigns specific RGB color channels to different parts of the aircraft (e.g., Head, Wings, Tail) so the posture changes are baked into the color gradients of the generated images.
* **Model Trainer (`CNN.py`):** Trains a Convolutional Neural Network (CNN) on these generated images. The script automatically evaluates categorical accuracy, saves the best-performing model (`.h5` file), and generates performance graphs comparing training and validation loss/accuracy.



