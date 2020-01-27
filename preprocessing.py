import numpy as np
import os
from sklearn.model_selection import train_test_split

path      = 'C:/Users/chenh/Desktop/Seg/'
img_path  = path + 'Image/'
msk_path  = path + 'Mask/'
data_name = list()
for img_name in os.listdir(img_path):
    data_name.append(img_name)
    data_name_np = np.array(data_name)
    
    
print(type(data_name_np))

left , test , mleft , _ = train_test_split(data_name_np, 
                                          np.ones_like(data_name_np),
                                          test_size=0.1)
train , valid , _ , _ = train_test_split(left, 
                                         mleft, 
                                         test_size=0.07 / (1 - 0.15))
print(train)
print(valid)
print(test)

print(len(train))
print(len(valid))
print(len(test))

np.save(os.path.join(path,"train_names.npy"),train)
np.save(os.path.join(path,"test_names.npy"),test)
np.save(os.path.join(path,"valid_names.npy"),valid)