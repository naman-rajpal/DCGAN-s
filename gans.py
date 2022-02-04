from __future__ import print_function
import torch
from skimage import io, transform
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


batchSize = 64
imageSize = 64

transform = transforms.Compose([transforms.CenterCrop([32,32]),transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform =None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__ (self):
        path, dirs, files = next(os.walk(self.root_dir))
        file_count = len(files)
        return(file_count-1)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        image_name_str = str(idx)+".jpg"
        image_name = os.path.join(self.root_dir,image_name_str)
        image = io.imread(image_name)
        
        sample = image
        if self.transform:
            sample = Image.fromarray(image.astype('uint8'), 'RGB')
            sample =self.transform(sample)
            
        return sample
        
          

    
dataset = CustomDataset(root_dir = 'images', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True) 
       
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    
    def forward(self,input):
        output = self.main(input)
        return output

netG = G()
netG.apply(weights_init)

class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3 ,64 ,4 ,2 ,1 ,bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )
    
    def forward(self,input):
        output = self.main(input)
        return output.view(-1)
    
netD = D()
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(),lr=0.0002,betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(),lr=0.0002,betas = (0.5, 0.999))

for epoch in range(5000):
    for i , data in enumerate(dataloader,0):
        netD.zero_grad()
        real = data
        input = Variable(real)
        output = netD(input)
        target = Variable(torch.ones(input.size()[0]))
        errD_real = criterion(output,target)
        
        noise = Variable(torch.randn(input.size()[0],100,1,1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output,target)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output,target)
        
        errG.backward()
        optimizerG.step()
        
        print("---------------------------")
        print("Epoch: "+str(epoch)+" err_G: "+str(errG.view(-1))+" err_D: "+str(errD.view(-1)))
        vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
