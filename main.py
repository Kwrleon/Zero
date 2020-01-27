vatimport os
import sys
import numpy as np
import torch
import torch.nn
from dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from model import UNet as model
from losses import tversky_loss, tversky_coeff


#选择设备
os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS" #设备排序
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' #设置第一块GPU可见
device = torch.device("cpu")#"cuda:0,1" if torch.cuda.is_available() else "cpu")
print(device)

#设置参数
batch_size = 2
v_batch_size = 2
epochs     = 50
lr         = 1e-3
print(f"Batch size: {batch_size}, LR: {lr}")

#设置路径
path     = 'C:/Users/chenh/Desktop/Seg/'
img_path = path + 'Image/'
msk_path = path + 'Mask/'
chk_path = path + 'log/checkpoints/'

#载入数据
transform = transforms.Compose([transforms.Resize(160),
                                transforms.ToTensor()])

train_name = np.load(os.path.join(path,"train_names.npy"))
valid_name = np.load(os.path.join(path,"valid_names.npy"))

train_set = Dataset(path = path, data_name = train_name, transform = transform)
valid_set = Dataset(path = path, data_name = valid_name, transform = transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=v_batch_size, shuffle=False)

#载入模型
model = model()
#summary(model, (1,400,640),device='cpu')
model = model.to(device)
#model = torch.nn.DataParallel(model)#GPU并行运算

#损失函数
criterion = torch.nn.BCELoss().to(device)
#定义优化器
optimizer = torch.optim.SGD(model.parameters(),
                            lr = lr,
                            momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones = [8,14,19,25,30,35],
                            gamma = 0.1)
print("Optimizer: ", optimizer.__class__.__name__)

#开始训练
print("Start Training")
for epoch in range(1,epochs+1):
    print("################## EPOCH {}/{} ##################".format(epoch, epochs))
    
    scheduler.step()
        
    for t_index, (img, msk) in enumerate(train_loader):
        model.train()
        model.zero_grad()
        img,msk = img.to(device),msk.to(device)
        out = model(img)
        loss = criterion(out,msk)
        optimizer.zero_grad()
        loss.backward
        optimizer.step()
        sys.stdout.write("\r[Train] [Epoch {}/{}] [Batch {}/{}] [Loss:{}] [Learning Rate:{}]".format(epoch,epochs,t_index+1,len(train_loader),loss.item(),optimizer.param_groups[0]['lr']))
        sys.stdout.flush()
        
        
        
 




