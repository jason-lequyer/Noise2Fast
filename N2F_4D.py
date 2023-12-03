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
            x = self.conv6(x)
            return x
        
        
    #load data
    file_list = [f for f in os.listdir(folder)]
    for v in range(len(file_list)):
        
        file_name =  file_list[v]
        if file_name[0] == '.':
            continue
        print(file_name)
        
        
        inp = imread(folder + '/' + file_name)
        if inp.shape[-1] == 3:
            inp = np.swapaxes(inp, -2, -1)
            inp = np.swapaxes(inp, -3, -2)
        start_time = time.time()
        ogshape = inp.shape

        inp = inp.reshape(-1,ogshape[-2],ogshape[-1])
        out = np.zeros(inp.shape, dtype=np.float32)
    
        for oz in range(inp.shape[0]):
            print('Slice '+str(oz+1)+'/'+str(inp.shape[0]))
            
            
            notdone = True
            learning_rate = learning_rate = 0.001
            while notdone:
                img = inp[oz,:,:]
                typer = type(inp[0,0,0])
                minner = np.mean(img)
                img = img - minner
                maxer = np.std(img)
                if maxer == 0:
                    goodo = False
                    notdone = False
                    H2 = img + minner
                    out[oz,:,:] = H2
                    continue
                
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
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                
                
                
                running_loss1=0.0
                running_loss2=0.0
                maxpsnr = -np.inf
                timesince = 0
                last10 = [0]*105
                last10psnr = [0]*105
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
                out[oz,:,:] = H.copy()
                torch.cuda.empty_cache()
                if np.sum(np.round(H[1:-1,1:-1]-np.mean(H[1:-1,1:-1]))>0) <= 25 and learning_rate != 0.000005:
                    learning_rate = 0.000005
                    print("Reducing learning rate")
                else:
                    notdone = False
                    print("--- %s seconds ---" % (time.time() - start_time))
                    start_time = time.time()
        out = out.reshape(ogshape)
            
            
        try:
            imwrite(outfolder + '/' + file_name, np.round(out).astype(typer), imagej=True)
        except:
            imwrite(outfolder + '/' + file_name, np.round(out).astype(np.float32), imagej=True)
