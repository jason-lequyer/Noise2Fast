import argparse
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import fnmatch
import sys
from tifffile import imread, imwrite
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
import time

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder = sys.argv[1]+'/'
    outdir = folder[:-1]+'_Ne2Ne/'
    Path(outdir).mkdir(exist_ok=True)
    
    class UNet(nn.Module):
      def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
    
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2", kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
    
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
    
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
    
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
    
        self.conv3 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
    
      def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
    
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        
        diffY = enc4.size()[2] - dec4.size()[2]
        diffX = enc4.size()[3] - dec4.size()[3]
        dec4 = F.pad(dec4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        
        diffY = enc3.size()[2] - dec3.size()[2]
        diffX = enc3.size()[3] - dec3.size()[3]
        dec3 = F.pad(dec3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        
        diffY = enc2.size()[2] - dec2.size()[2]
        diffX = enc2.size()[3] - dec2.size()[3]
        dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        
        diffY = enc1.size()[2] - dec1.size()[2]
        diffX = enc1.size()[3] - dec1.size()[3]
        dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = torch.sigmoid(
                  self.conv(dec1)   
              )
        return out
    
      @staticmethod
      def _block(in_channels, features, name, kernel_size=3, padding=1):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    class CustomImageDataset(Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.LEN = len( fnmatch.filter(os.listdir(img_dir), '*.tif') )
            print(self.LEN)
    
        def __len__(self):
            return self.LEN
    
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir,  str(idx) + '.tif')
            image = imread(img_path)
            minner = np.amin(image)
            image = image-minner
            maxer = np.amax(image)
            image = image/maxer
            image = torch.from_numpy(image)
            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image,0)
            return (image, '')
    
    
    
    
    
    class Neighbour2Neighbour():
        def __init__(self, gamma=2, k=2):
            self.gamma = gamma
            self.k = k
            self.LR = 0.0003
            self.transforms = transforms.Compose(
                [transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))])
            self.use_cuda = torch.cuda.is_available()
    
        def __get_args__(self):
            parser = argparse.ArgumentParser(description='Parameters')
            parser.add_argument('--epochs', type=int, default=15)
            parser.add_argument('--batch', type=int, default=4)
            parser.add_argument('--var', type=float, default=.5)
            parser.add_argument('--learning_rate', type=float, default=.0005)
            parser.add_argument('--data_dir', type=str, default='./data')
            parser.add_argument('--checkpoint_dir', type=str,
                                default='./checkpoints')
    
            args = parser.parse_args()
            return (args.epochs, args.batch, args.var, args.learning_rate, args.data_dir, args.checkpoint_dir)
    
        def subsample(self, image):
            # This function only works for k = 2 as of now.
            blength, channels, m, n = np.shape(image)
            dim1, dim2 = m // self.k, n // self.k
            image1, image2 = np.zeros([blength, channels, dim1, dim2]), np.zeros(
                [blength, channels, dim1, dim2])
    
            image_cpu = image.cpu()
            for blen in range(blength):
                for channel in range(channels):
                    for i in range(dim1):
                        for j in range(dim2):
                            i1 = i * self.k
                            j1 = j * self.k
                            num = np.random.choice([0, 1, 2, 3])
                            if num == 0:
                                image1[blen, channel, i, j], image2[blen, channel, i, j] = image_cpu[blen,
                                                                                               channel, i1, j1], image_cpu[blen, channel, i1, j1+1]
                            elif num == 1:
                                image1[blen, channel, i, j], image2[blen, channel, i, j] = image_cpu[blen,
                                                                                               channel, i1+1, j1], image_cpu[blen, channel, i1+1, j1+1]
                            elif num == 2:
                                image1[blen, channel, i, j], image2[blen, channel, i, j] = image_cpu[blen,
                                                                                               channel, i1, j1], image_cpu[blen, channel, i1+1, j1]
                            else:
                                image1[blen, channel, i, j], image2[blen, channel, i, j] = image_cpu[blen,
                                                                                               channel, i1, j1+1], image_cpu[blen, channel, i1+1, j1+1]
    
            if self.use_cuda:
                return torch.from_numpy(image1).cuda(), torch.from_numpy(image2).cuda()
            return torch.from_numpy(image1).double(), torch.from_numpy(image2).double()
    
    
        def get_model(self):
            model = UNet(in_channels=1, out_channels=1).double()
            if self.use_cuda:
                model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
            criterion = RegularizedLoss()
            return model, [], optimizer, criterion
    
        def train(self, imgin):
            model, noisy, optimizer, criterion = self.get_model()
            if self.use_cuda:
                model = model.cuda()
    
            for epoch in range(1,101):
                total_loss_valid = 0
                total_loss = 0
                optimizer.zero_grad()
                noisy_image = imgin
    
                g1, g2 = self.subsample(noisy_image)
                fg1 = model(g1)
                with torch.no_grad():
                    X = model(noisy_image)
                    G1, G2 = self.subsample(X)
                
                total_loss = criterion(fg1, g2, G1, G2)
                total_loss.backward()
                optimizer.step()
    
    
                if epoch % 20 == 0:
                    imwrite(str(epoch)+'.tif', ((maxer*X.detach().cpu().numpy())+minner)[0,0,:,:])
            imwrite(outdir+file, ((maxer*X.detach().cpu().numpy())+minner)[0,0,:,:])
                    
        

    
    
    class RegularizedLoss(nn.Module):
        def __init__(self, gamma=2):
            super().__init__()
    
            self.gamma = gamma
    
        def mseloss(self, image, target):
            x = ((image - target)**2)
            return torch.mean(x)
    
        def regloss(self, g1, g2, G1, G2):
            return torch.mean((g1-g2-G1+G2)**2)
    
        def forward(self, fg1, g2, G1f, G2f):
            return self.mseloss(fg1, g2) + self.gamma * self.regloss(fg1, g2, G1f, G2f)
    
    
    N2N = Neighbour2Neighbour(gamma=2)
    
    for file in os.listdir(folder):
        start_time = time.time()
        print(file)
        image = imread(folder+file)
        minner = np.amin(image)
        image = image-minner
        maxer = np.amax(image)
        image = image/maxer
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image,0)
        image = image.to(device)
        N2N.train(image)
        print("--- %s seconds ---" % (time.time() - start_time))
