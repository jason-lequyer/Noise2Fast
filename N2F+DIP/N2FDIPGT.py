
  
#Copyright 2021, Jason Lequyer and Laurence Pelletier, All rights reserved.
#Sinai Health SystemLunenfeld-Tanenbaum Research Institute
#600 University Avenue, Room 1070
#Toronto, ON, M5G 1X5, Canada
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tifffile import imread, imwrite
import sys
import torch.utils.data as utils_data
import numpy as np
from pathlib import Path
import time
from skimage.metrics import peak_signal_noise_ratio as psnr


if __name__ == "__main__":
    tsince = 100
    folder = sys.argv[1]
    outfolder = folder+'_N2F'
    Path(outfolder).mkdir(exist_ok=True)
    
    p2d = (0, 1, 0, 1) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class TwoCon(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.batch1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.batch2 = nn.BatchNorm2d(out_channels)
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(self.batch1(x))
            x = self.conv2(x)
            x = F.relu(self.batch2(x))
            return x
        
    
    class Down(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv = TwoCon(in_channels,out_channels)
    
        def forward(self, x):
            x = self.pool(x)
            x = self.conv(x)
            return x
    
    class Up(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.upconv = nn.ConvTranspose2d(in_channels,out_channels,2,2)
            self.conv = TwoCon(in_channels,out_channels)
    
        def forward(self, x, y):
            x = self.upconv(x)
            if x.shape[2] != y.shape[2]:
                x = F.pad(x, p2d, "constant", 0)
            x = torch.cat([x, y], dim=1)
            x = self.conv(x)
            return x
    sizer = 2
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, sizer*8)
            self.conv2 = Down(sizer*8, sizer*16)
            self.conv3 = Down(sizer*16, sizer*32)
            self.conv4 = Down(sizer*32, sizer*64)
            self.conv5 = Down(sizer*64, sizer*128)   
            self.upconv1 = Up(sizer*128, sizer*64)
            self.upconv2 = Up(sizer*64, sizer*32)
            self.upconv3 = Up(sizer*32, sizer*16)
            self.upconv4 = Up(sizer*16, sizer*8)
            self.conv6 = nn.Conv2d(sizer*8,1,1)
            
    
        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x4 = self.conv4(x3)
            x = self.conv5(x4)
            x = self.upconv1(x,x4)
            x = self.upconv2(x,x3)
            x = self.upconv3(x,x2)
            x = self.upconv4(x,x1)
            x = F.sigmoid(self.conv6(x))
            return x

        
    file_list = [f for f in os.listdir(folder)]
    start_time = time.time()
    for v in range(len(file_list)):
        
        file_name =  file_list[v]
        print(file_name)
        if file_name[0] == '.':
            continue
        
        img = imread(folder + '/' + file_name)
        typer = type(img[0,0])
        
        minner = np.amin(img)
        img = img - minner
        maxer = np.amax(img)
        img = img/maxer
        img = img.astype(np.float32)
        shape = img.shape
        
        GTo = imread(folder + '/' + file_name)
        
        
    
        listimgH = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin.shape[0]):
            for j in range(imgin.shape[1]):
                if j % 2 == 0:
                    imgin[i,j] = imgZ[2*i+1,j]
                    imgin2[i,j] = imgZ[2*i,j]
                if j % 2 == 1:
                    imgin[i,j] = imgZ[2*i,j]
                    imgin2[i,j] = imgZ[2*i+1,j]
        imgin = torch.from_numpy(imgin)
        imgin = torch.unsqueeze(imgin,0)
        imgin = torch.unsqueeze(imgin,0)
        imgin = imgin.to(device)
        imgin2 = torch.from_numpy(imgin2)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = imgin2.to(device)
        listimgH.append(imgin)
        listimgH.append(imgin2)
        
        listimgV = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin3 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin3.shape[0]):
            for j in range(imgin3.shape[1]):
                if i % 2 == 0:
                    imgin3[i,j] = imgZ[i,2*j+1]
                    imgin4[i,j] = imgZ[i, 2*j]
                if i % 2 == 1:
                    imgin3[i,j] = imgZ[i,2*j]
                    imgin4[i,j] = imgZ[i,2*j+1]
        imgin3 = torch.from_numpy(imgin3)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = imgin3.to(device)
        imgin4 = torch.from_numpy(imgin4)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = imgin4.to(device)
        listimgV.append(imgin3)
        listimgV.append(imgin4)
        
    
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(device)
        
        listimgV1 = [[listimgV[0],listimgV[1]]]
        listimgV2 = [[listimgV[1],listimgV[0]]]
        listimgH1 = [[listimgH[1],listimgH[0]]]
        listimgH2 = [[listimgH[0],listimgH[1]]]
        listimg = listimgH1+listimgH2+listimgV1+listimgV2
        
        net = Net()
        net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.03)
        
        
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*100
        last10psnr = [0]*100
        cleaned = 0
        while timesince <= tsince:
            indx = np.random.randint(0,len(listimg))
            data = listimg[indx]
            inputs = data[0]
            labello = data[1]
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss1 = criterion(outputs, labello)
            loss = loss1
            running_loss1+=loss1.item()
            loss.backward()
            optimizer.step()
            
            
            running_loss1=0.0
            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned*maxer+minner)
                outputstest = net(img)
                cleaned = outputstest[0,0,:,:].cpu().detach().numpy()
                
                noisy = img.cpu().detach().numpy()
                ps = -np.mean(((GTo-minner)/maxer-cleaned)**2)
                last10psnr.pop(0)
                last10psnr.append(ps)
                if ps > maxpsnr:
                    #psGT = psnr(GTo,cleaned*maxer+minner,data_range = 255)
                    #print(ps)
                    #print(psGT)
                    maxpsnr = ps
                    outclean = cleaned*maxer+minner
                    timesince = 0
                else:
                    timesince+=1.0
                        
        
        H = outclean
        #psGT = psnr(GTo,H,data_range = 255)
        #print("--- %s seconds ---" % (time.time() - start_time))
        #print(psGT)
        #sys.exit()
        
        imwrite(outfolder + '/' + file_name, H.astype(typer))
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        
        
        torch.cuda.empty_cache()
    
