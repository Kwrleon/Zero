import os
import torch
from PIL import Image
 
class Dataset(torch.utils.data.Dataset):
    def __init__(self , path , data_name , transform):
        img_path = os.path.join(path,"Image")
        msk_path = os.path.join(path,"Mask")
        data_name = data_name
        samples = []
        for name in data_name:
            img = img_path + "/" + name
            msk = msk_path + "/" + name
            sample = (img,msk)
            samples.append(sample)
        self.samples = samples
        self.transform = transform
    def __getitem__(self,index):
        img_n,msk_n = self.samples[index]
        print(index)
        img = Image.open(img_n).convert('L')
        msk = Image.open(msk_n).convert('1') 
    
        img = self.transform(img)
        msk = self.transform(msk)        
        return [img,msk]
        
    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import numpy as np
    from torchvision import transforms
    path     = '../'
    train_name = np.load(os.path.join(path,"train_names.npy"))
    valid_name = np.load(os.path.join(path,"valid_names.npy"))
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = Dataset(path = path, data_name = train_name, transform = transform)
    valid_set = Dataset(path = path, data_name = valid_name, transform = transform)