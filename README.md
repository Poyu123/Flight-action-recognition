# Flight-action-recognition
This proj. introduces two db construction methods and recognition methods, achieving fast and accurate aircraft movement recognition. It starts with flight segment machine ID, then integrates labelled aircraft fragments for training and verification db construction, and trains FCNN and CNN. The method exhibits excellent recognition capabilities.

# Preface
This work is really interesting but a bit challenging. Anyway we did a lot of things and these potentially give you a hand. First of all, I would like to declare that all of data we used are from a nonearning cloud belonged to Glowlingsever. Here is its website address : https://drive.google.com/drive/folders/178nooqDx1yNS9VZ01ioOjfjEXe_AC1E9. And those data really play a significant role in our work since it is not necessary to fly a plane on our own! We took advantage of around 50Gb to do this research. This means that you'd better to follow this way to save time.

# 00
## Download
I will introduce the code file 'downloadpresolve.py', let you know how this file functions and simply talk about its purpose. And please remember that this type of method is also utilised in the left '.py' files.
First, you need to download such a 'raw' document ending with **'.zip.acmi'** through the above website mentioned in Preface. Then, change the addresses in downloadpresolve.py of two variables named **'download_dir' and 'output_dir'** into yours. Please be careful since these two addresses must ***exist*** and they are only two ***folders***.  If you succeed, there will be such a file ending only with **'.acmi'**. And this is what we truely need. This .py file achieves the action of decompressing. One more important thing is that you'd better **backup file** to lessen the adverse impacts like computer crash though there is only slight risk.  
