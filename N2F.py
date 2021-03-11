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


if __name__ == "__main__":
    tsince = 100
    folder = sys.argv[1]
    outfolder = folder+'_N2F'
    Path(outfolder).mkdir(exist_ok=True)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class TwoCon(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            return x
    
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, 64)
            self.conv2 = TwoCon(64, 64)
            self.conv3 = TwoCon(64, 64)
            self.conv4 = TwoCon(64, 64)  
            self.conv6 = nn.Conv2d(64,1,1)
            
    
        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x = self.conv4(x3)
            x = torch.sigmoid(self.conv6(x))
            return x
        
        
    #load data
    file_list = [f for f in os.listdir(folder)]
    for v in range(len(file_list)):
        start_time = time.time()
        file_name =  file_list[v]
        print(file_name)
        
        net = Net()
        net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
        img = imread(folder + '/' + file_name)
        
        minner = np.amin(img)
        img = img - minner
        maxer = np.amax(img)
        img = img/maxer
        img = img.astype(np.float32)
        shape = img.shape
        
        
    
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
        listimgH.append(imgin.copy())
        listimgH.append(imgin2.copy())
        
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
        listimgV.append(imgin3.copy())
        listimgV.append(imgin4.copy())
        
    
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(device)
        
        listimgV1 = [[listimgV[0],listimgV[1]]]
        listimgV2 = [[listimgV[1],listimgV[0]]]
        listimgH1 = [[listimgH[1],listimgH[0]]]
        listimgH2 = [[listimgH[0],listimgH[1]]]
        listimg = listimgH1+listimgH2+listimgV1+listimgV2
        
        class MyDataSet(utils_data.Dataset):
          def __init__(self, transform=None):
              self.image_pair_list = listimg
              
          def __len__(self):
              return len(self.image_pair_list)
          
          def __getitem__(self, index):
              image_pair =  self.image_pair_list[index]
              
              
              mask = image_pair[0]
              label = image_pair[1]
              mask = torch.from_numpy(mask)
              mask = torch.unsqueeze(mask,0)
              label = torch.from_numpy(label)
              label = torch.unsqueeze(label,0)
              
              
              
              return mask, label
        
        trainset = MyDataSet()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
        
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*105
        last10psnr = [0]*105
        cleaned = 0
        while timesince <= tsince:
    
            for i, data in enumerate(trainloader):
                inputs, labello = data[0].to(device), data[1].to(device)
                
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
                    ps = -np.mean((noisy-cleaned)**2)
                    last10psnr.pop(0)
                    last10psnr.append(ps)
                    if ps > maxpsnr:
                        maxpsnr = ps
                        outclean = cleaned*maxer+minner
                        timesince = 0
                    else:
                        timesince+=1.0
                        
        
        H = np.mean(last10, axis=0)
        
        imwrite(outfolder + '/' + file_name, H)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        
        
        torch.cuda.empty_cache()
