import torch
from torch import nn
from tifffile import imread, imwrite
import numpy as np
import sys
import torch.nn.functional as F

psize = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
inimg = imread('just11/11.tif').astype(np.float32)
inimg2 = imread('just11/12.tif').astype(np.float32)
inimg3 = imread('just11/10.tif').astype(np.float32)
inimg4 = imread('just11/09.tif').astype(np.float32)




shape = inimg.shape

Zshape = [shape[0],shape[1]]
if shape[0] % 2 == 1:
    Zshape[0] -= 1
if shape[1] % 2 == 1:
    Zshape[1] -=1  
imgZ = inimg[:Zshape[0],:Zshape[1]]

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


Zshape = [shape[0],shape[1]]
if shape[0] % 2 == 1:
    Zshape[0] -= 1
if shape[1] % 2 == 1:
    Zshape[1] -=1  
imgZ = inimg[:Zshape[0],:Zshape[1]]

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

inimg = torch.from_numpy(inimg)
inimg = torch.unsqueeze(inimg,0)
inimg = torch.unsqueeze(inimg,0)

inimg = inimg.to(device)

inimg2 = torch.from_numpy(inimg2)
inimg2 = torch.unsqueeze(inimg2,0)
inimg2 = torch.unsqueeze(inimg2,0)
inimg2 = inimg2.to(device)

inimg3 = torch.from_numpy(inimg3)
inimg3 = torch.unsqueeze(inimg3,0)
inimg3 = torch.unsqueeze(inimg3,0)
inimg3 = inimg3.to(device)

inimg4 = torch.from_numpy(inimg4)
inimg4 = torch.unsqueeze(inimg4,0)
inimg4 = torch.unsqueeze(inimg4,0)
inimg4 = inimg4.to(device)


unfold = nn.Unfold(psize)
fold = nn.Fold(output_size=(512, 512), kernel_size=psize)

print(inimg.shape)

RP = nn.ReflectionPad2d(1)


inimgexpand = RP(inimg)

inimgleft = inimgexpand[:,:,2:,1:-1]
inimgright = inimgexpand[:,:,:-2,1:-1]
inimgup = inimgexpand[:,:,1:-1,:-2]
inimgdown = inimgexpand[:,:,1:-1,2:]

inimgleft = (inimg - inimgleft)**2
inimgright = (inimg - inimgright)**2
inimgup = (inimg - inimgup)**2
inimgdown = (inimg - inimgdown)**2

inimgavg = (inimgleft + inimgright + inimgup + inimgdown)/4
inimgavg = inimgavg.to(device)

inimg = unfold(inimg)
inimg = torch.swapaxes(inimg,1,2)

inimg2 = unfold(inimg2)
inimg2 = torch.swapaxes(inimg2,1,2)

inimg3 = unfold(inimg3)
inimg3 = torch.swapaxes(inimg3,1,2)

inimg4 = unfold(inimg4)
inimg4 = torch.swapaxes(inimg4,1,2)

inimgavg = unfold(inimgavg)
inimgavg = torch.swapaxes(inimgavg,1,2)
inimgavg = torch.mean(inimgavg[0,:,:],axis=1)




imgin = unfold(imgin)
imgin = torch.swapaxes(imgin,1,2)

imgin2 = unfold(imgin2)
imgin2 = torch.swapaxes(imgin2,1,2)

imgin3 = unfold(imgin3)
imgin3 = torch.swapaxes(imgin3,1,2)

imgin4 = unfold(imgin4)
imgin4 = torch.swapaxes(imgin4,1,2)

imgin = imgin - torch.unsqueeze(torch.mean(imgin, axis=2),2)
imgin2 = imgin2 - torch.unsqueeze(torch.mean(imgin2, axis=2),2)
imgin3 = imgin3 - torch.unsqueeze(torch.mean(imgin3, axis=2),2)
imgin4 = imgin4 - torch.unsqueeze(torch.mean(imgin4, axis=2),2)

inimg = inimg - torch.unsqueeze(torch.mean(inimg, axis=2),2)
inimg2 = inimg2 - torch.unsqueeze(torch.mean(inimg2, axis=2),2)
inimg3 = inimg3 - torch.unsqueeze(torch.mean(inimg3, axis=2),2)
inimg4 = inimg4 - torch.unsqueeze(torch.mean(inimg4, axis=2),2)

inimgavg = torch.ones_like(inimgavg).to(device)
inimgavg = 2*inimgavg




def avgsimpatches(smallimg,bigimg,avgimg):
    counts = torch.zeros_like(smallimg[0,:,0])
    for i in range(smallimg.shape[1]):
        if i % 10000 == 0:
            print(i)
        diff = torch.mean((smallimg[0,i:i+1,:] - bigimg[0,:,:])**2, axis=1)
        diff = avgimg - diff
        diff = torch.clip(diff,0,1)
        counts[i] = torch.count_nonzero(diff)
    return torch.mean(counts)
    

print('Image 11: '+str(avgsimpatches(inimg,inimg,inimgavg)))
print('Odd-left: '+str(avgsimpatches(imgin,inimg,inimgavg)))
print('Even-left: '+str(avgsimpatches(imgin2,inimg,inimgavg)))
print('Odd-up: '+str(avgsimpatches(imgin3,inimg,inimgavg)))
print('Even-up: '+str(avgsimpatches(imgin4,inimg,inimgavg)))
print('Image 12: '+str(avgsimpatches(inimg2,inimg,inimgavg)))
print('Image 10: '+str(avgsimpatches(inimg3,inimg,inimgavg)))
print('Image 9: '+str(avgsimpatches(inimg4,inimg,inimgavg)))

sys.exit()
    





