import torch

class Conv_block(torch.nn.Module):
    def __init__(self,im_ch, ot_ch):
        super(Conv_block,self).__init__()
        self.Conv_block = torch.nn.Sequential(
                torch.nn.Conv2d(im_ch, ot_ch,kernel_size=3,stride=1,padding=1,bias=True),
                torch.nn.BatchNorm2d(ot_ch),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(ot_ch, ot_ch,kernel_size=3,stride=1,padding=1,bias=True),
                torch.nn.BatchNorm2d(ot_ch),
                torch.nn.ReLU(inplace=True),
                )
    def forward(self,x):
        x = self.Conv_block(x)
        return x
    
class Up_Conv(torch.nn.Module):
    def __init__(self,im_ch, ot_ch):
        super(Up_Conv,self).__init__()
        self.Up_Conv = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(im_ch,ot_ch,kernel_size=3,stride=1,padding=1,bias=True),
                torch.nn.BatchNorm2d(ot_ch),
                torch.nn.ReLU(inplace=True)
                )
    def forward(self,x):
        x = self.Up_Conv(x)
        return x
        
class UNet(torch.nn.Module):
    def __init__(self, import_ch = 1, output_ch = 1):
        super(UNet,self).__init__()
        self.Maxpool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Conv1 = Conv_block(im_ch = import_ch, ot_ch = 64)
        self.Conv2 = Conv_block(im_ch = 64, ot_ch = 128)
        self.Conv3 = Conv_block(im_ch = 128, ot_ch = 256)
        self.Conv4 = Conv_block(im_ch = 256, ot_ch = 512)
        self.Conv5 = Conv_block(im_ch = 512, ot_ch = 1024)
        
        self.Up5  = Up_Conv(im_ch = 1024, ot_ch = 512)
        self.UpC5 = Conv_block(im_ch = 1024, ot_ch = 512)
        self.Up4  = Up_Conv(im_ch = 512, ot_ch = 256)
        self.UpC4 = Conv_block(im_ch = 512, ot_ch = 256)
        self.Up3  = Up_Conv(im_ch = 256, ot_ch = 128)
        self.UpC3 = Conv_block(im_ch = 256, ot_ch = 128)
        self.Up2  = Up_Conv(im_ch = 128, ot_ch = 64)
        self.UpC2 = Conv_block(im_ch = 128, ot_ch = 64)
        
        self.ConvZ =  torch.nn.Sequential(
                torch.nn.Conv2d(64,output_ch,
                                kernel_size=1,
                                stride=1,
                                padding=0),
                torch.nn.Sigmoid())
        
    def forward(self,x):
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  
        
        z5 = self.Up5(x5)
        z5 = torch.cat((x4,z5),dim=1)
        z5 = self.UpC5(z5)
        
        z4 = self.Up4(z5)
        z4 = torch.cat((x3,z4),dim=1)
        z4 = self.UpC4(z4)
        
        z3 = self.Up3(z4)
        z3 = torch.cat((x2,z3),dim=1)
        z3 = self.UpC3(z3)
        
        z2 = self.Up2(z3)
        z2 = torch.cat((x1,z2),dim=1)
        z2 = self.UpC2(z2)
        
        z1 = self.ConvZ(z2)
        
        return z1
        
        
        
            
        
        

        