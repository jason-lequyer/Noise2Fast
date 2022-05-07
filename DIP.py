import time

from pathlib import Path
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import torch
import torch.optim
from torch.optim import Adam
from torch.nn import MSELoss
import torch.nn as nn
import sys 
from tifffile import imread, imwrite


if __name__ == "__main__":
    folder = sys.argv[1]
    #folder = 'Set12_gaussian25'
    outfolder = folder+'_DIP'
    file_list = [f for f in os.listdir(folder)]
    Path(outfolder).mkdir(exist_ok=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    dtype = torch.cuda.FloatTensor
    
    imsize =-1
    PLOT = True
    
    class Downsampler(nn.Module):
        '''
            http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
        '''
        def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
            super(Downsampler, self).__init__()
            
            assert phase in [0, 0.5], 'phase should be 0 or 0.5'
    
            if kernel_type == 'lanczos2':
                support = 2
                kernel_width = 4 * factor + 1
                kernel_type_ = 'lanczos'
    
            elif kernel_type == 'lanczos3':
                support = 3
                kernel_width = 6 * factor + 1
                kernel_type_ = 'lanczos'
    
            elif kernel_type == 'gauss12':
                kernel_width = 7
                sigma = 1/2
                kernel_type_ = 'gauss'
    
            elif kernel_type == 'gauss1sq2':
                kernel_width = 9
                sigma = 1./np.sqrt(2)
                kernel_type_ = 'gauss'
    
            elif kernel_type in ['lanczos', 'gauss', 'box']:
                kernel_type_ = kernel_type
    
            else:
                assert False, 'wrong name kernel'
                
                
            # note that `kernel width` will be different to actual size for phase = 1/2
            self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
            
            downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
            downsampler.weight.data[:] = 0
            downsampler.bias.data[:] = 0
    
            kernel_torch = torch.from_numpy(self.kernel)
            for i in range(n_planes):
                downsampler.weight.data[i, i] = kernel_torch       
    
            self.downsampler_ = downsampler
    
            if preserve_size:
    
                if  self.kernel.shape[0] % 2 == 1: 
                    pad = int((self.kernel.shape[0] - 1) / 2.)
                else:
                    pad = int((self.kernel.shape[0] - factor) / 2.)
                    
                self.padding = nn.ReplicationPad2d(pad)
            
            self.preserve_size = preserve_size
            
        def forward(self, input):
            if self.preserve_size:
                x = self.padding(input)
            else:
                x= input
            self.x = x
            return self.downsampler_(x)
    
    def add_module(self, module):
        self.add_module(str(len(self) + 1), module)
        
    torch.nn.Module.add = add_module
    
    class Concat(nn.Module):
        def __init__(self, dim, *args):
            super(Concat, self).__init__()
            self.dim = dim
    
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    
        def forward(self, input):
            inputs = []
            for module in self._modules.values():
                inputs.append(module(input))
    
            inputs_shapes2 = [x.shape[2] for x in inputs]
            inputs_shapes3 = [x.shape[3] for x in inputs]        
    
            if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
                inputs_ = inputs
            else:
                target_shape2 = min(inputs_shapes2)
                target_shape3 = min(inputs_shapes3)
    
                inputs_ = []
                for inp in inputs: 
                    diff2 = (inp.size(2) - target_shape2) // 2 
                    diff3 = (inp.size(3) - target_shape3) // 2 
                    inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
    
            return torch.cat(inputs_, dim=self.dim)
    
        def __len__(self):
            return len(self._modules)
    
    
    class GenNoise(nn.Module):
        def __init__(self, dim2):
            super(GenNoise, self).__init__()
            self.dim2 = dim2
    
        def forward(self, input):
            a = list(input.size())
            a[1] = self.dim2
            # print (input.data.type())
    
            b = torch.zeros(a).type_as(input.data)
            b.normal_()
    
            x = torch.autograd.Variable(b)
    
            return x
    
    
    class Swish(nn.Module):
        """
            https://arxiv.org/abs/1710.05941
            The hype was so huge that I could not help but try it
        """
        def __init__(self):
            super(Swish, self).__init__()
            self.s = nn.Sigmoid()
    
        def forward(self, x):
            return x * self.s(x)
    
    
    def act(act_fun = 'LeakyReLU'):
        '''
            Either string defining an activation function or module (e.g. nn.ReLU)
        '''
        if isinstance(act_fun, str):
            if act_fun == 'LeakyReLU':
                return nn.LeakyReLU(0.2, inplace=True)
            elif act_fun == 'Swish':
                return Swish()
            elif act_fun == 'ELU':
                return nn.ELU()
            elif act_fun == 'none':
                return nn.Sequential()
            else:
                assert False
        else:
            return act_fun()
    
    
    def bn(num_features):
        return nn.BatchNorm2d(num_features)
    
    
    def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
        downsampler = None
        if stride != 1 and downsample_mode != 'stride':
    
            if downsample_mode == 'avg':
                downsampler = nn.AvgPool2d(stride, stride)
            elif downsample_mode == 'max':
                downsampler = nn.MaxPool2d(stride, stride)
            elif downsample_mode  in ['lanczos2', 'lanczos3']:
                downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
            else:
                assert False
    
            stride = 1
    
        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            padder = nn.ReflectionPad2d(to_pad)
            to_pad = 0
      
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    
    
        layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
        return nn.Sequential(*layers)
    
    
    
    def skip(
            num_input_channels=2, num_output_channels=3, 
            num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
            filter_size_down=3, filter_size_up=3, filter_skip_size=1,
            need_sigmoid=True, need_bias=True, 
            pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
            need1x1_up=True):
        """Assembles encoder-decoder with skip connections.
    
        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
    
        """
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    
        n_scales = len(num_channels_down) 
    
        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
            upsample_mode   = [upsample_mode]*n_scales
    
        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            downsample_mode   = [downsample_mode]*n_scales
        
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
            filter_size_down   = [filter_size_down]*n_scales
    
        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
            filter_size_up   = [filter_size_up]*n_scales
    
        last_scale = n_scales - 1 
    
        cur_depth = None
    
        model = nn.Sequential()
        model_tmp = model
    
        input_depth = num_input_channels
        for i in range(len(num_channels_down)):
    
            deeper = nn.Sequential()
            skip = nn.Sequential()
    
            if num_channels_skip[i] != 0:
                model_tmp.add(Concat(1, skip, deeper))
            else:
                model_tmp.add(deeper)
            
            model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
    
            if num_channels_skip[i] != 0:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                skip.add(bn(num_channels_skip[i]))
                skip.add(act(act_fun))
                
            # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))
    
            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))
    
            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))
    
            deeper_main = nn.Sequential()
    
            if i == len(num_channels_down) - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                deeper.add(deeper_main)
                k = num_channels_up[i + 1]
    
            deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
    
            model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))
    
    
            if need1x1_up:
                model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                model_tmp.add(bn(num_channels_up[i]))
                model_tmp.add(act(act_fun))
    
            input_depth = num_channels_down[i]
            model_tmp = deeper_main
    
        model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            model.add(nn.Sigmoid())
    
        return model
    
    def fill_noise(x, noise_type):
        """Fills tensor `x` with noise of type `noise_type`."""
        if noise_type == 'u':
            x.uniform_()
        elif noise_type == 'n':
            x.normal_() 
        else:
            assert False
    
    def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
        """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
        """
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if method == 'noise':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
            net_input = torch.zeros(shape)
            
            fill_noise(net_input, noise_type)
            net_input *= var            
        elif method == 'meshgrid': 
            assert input_depth == 2
            X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
            meshgrid = np.concatenate([X[None,:], Y[None,:]])
            net_input=  torch.from_numpy(meshgrid)
        else:
            assert False
            
        return net_input
    
    def get_params(opt_over, net, net_input, downsampler=None):
        '''Returns parameters that we want to optimize over.
    
        Args:
            opt_over: comma separated list, e.g. "net,input" or "net"
            net: network
            net_input: torch.Tensor that stores input `z`
        '''
        opt_over_list = opt_over.split(',')
        params = []
        
        for opt in opt_over_list:
        
            if opt == 'net':
                params += [x for x in net.parameters() ]
            elif  opt=='down':
                assert downsampler is not None
                params = [x for x in downsampler.parameters()]
            elif opt == 'input':
                net_input.requires_grad = True
                params += [net_input]
            else:
                assert False, 'what is it?'
                
        return params
    
    def optimize(optimizer_type, parameters, closure, LR, num_iter):
        """Runs optimization loop.
    
        Args:
            optimizer_type: 'LBFGS' of 'adam'
            parameters: list of Tensors to optimize over
            closure: function, that returns loss variable
            LR: learning rate
            num_iter: number of iterations 
        """
        if optimizer_type == 'LBFGS':
            # Do several steps with adam first
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            for j in range(100):
                optimizer.zero_grad()
                closure()
                optimizer.step()
    
            print('Starting optimization with LBFGS')        
            def closure2():
                optimizer.zero_grad()
                return closure()
            optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
            optimizer.step(closure2)
    
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=LR)
            
            for j in range(num_iter):
                optimizer.zero_grad()
                closure()
                optimizer.step()
        else:
            assert False
        
    for v in range(len(file_list)):
        start_time = time.time()
        file_name =  file_list[v]
        print(file_name)
        noisy_image = imread(folder + '/' + file_name)
        minner = np.amin(noisy_image)
        noisy_image = noisy_image - minner
        maxer = np.amax(noisy_image)
        noisy_image = noisy_image/maxer
        
        img_noisy_np = np.expand_dims(noisy_image,0)
        
        INPUT = 'noise' # 'meshgrid'
        pad = 'reflect'
        OPT_OVER = 'net' # 'net,input'
        
        LR = 0.01
        
        OPTIMIZER='adam' # 'LBFGS'
        show_every = 100
        exp_weight=0.99
        
        num_iter = 3000
        input_depth = 1
        figsize = 5 
        
        net = skip(
                    input_depth, 1, 
                    num_channels_down = [8, 16, 32, 64, 128], 
                    num_channels_up   = [8, 16, 32, 64, 128],
                    num_channels_skip = [0, 0, 0, 4, 4], 
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        net = net.type(dtype)
        
        
            
        net_input = get_noise(input_depth, INPUT, (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()

        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
        
        # Loss
        mse = torch.nn.MSELoss().type(dtype)
        
        img_noisy_torch = torch.from_numpy(img_noisy_np).type(dtype)
        img_noisy_torch = torch.unsqueeze(img_noisy_torch,0)
        
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        out_avg = None
        last_net = None
        psrn_noisy_last = 0
        
        i = 0
        def closure():
            
            global i, out_avg, psrn_noisy_last, last_net, net_input
            
            out = net(net_input)
            
            # Smoothing
            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                    
            total_loss = mse(out, img_noisy_torch)
            total_loss.backward()
                
            
            psrn_noisy = np.mean((img_noisy_np - out.detach().cpu().numpy()[0])**2) 
    
            # Backtracking
            if i % show_every:
                if psrn_noisy - psrn_noisy_last < -5: 
                    print('Falling back to previous checkpoint.')
        
                    for new_param, net_param in zip(last_net, net.parameters()):
                        net_param.data.copy_(new_param.cuda())
        
                    return total_loss*0
                else:
                    last_net = [x.detach().cpu() for x in net.parameters()]
                    psrn_noisy_last = psrn_noisy
                    
            i += 1
            
            
            imwrite(outfolder + '/' + file_name, maxer*out[0,0,:,:].cpu().detach().numpy()+minner)
        
            return total_loss
        
        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)


        
    
        print("--- %s seconds ---" % (time.time() - start_time))
