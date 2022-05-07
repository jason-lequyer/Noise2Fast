import tensorflow as tf
import network.Punet
import numpy as np
from pathlib import Path
import util
import cv2
import os
import sys
from tifffile import imwrite
import tensorflow as tf
import numpy as np
from keras.utils import conv_utils
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 1000
N_STEP = 150000


if __name__ == "__main__":
    tsince = 100
    folder = sys.argv[1]
    outfolder = folder+'_quickshot'
    Path(outfolder).mkdir(exist_ok=True)
    
    class PConv2D(Conv2D):
        def __init__(self, *args, n_channels=3, mono=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
    
        def build(self, input_shape):
            """Adapted from original _Conv() layer of Keras        
            param input_shape: list of dimensions for [img, mask]
            """
    
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
    
            if input_shape[0][channel_axis] is None:
                raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
    
            self.input_dim = input_shape[0][channel_axis]
    
            # Image kernel
            kernel_shape = self.kernel_size + (self.input_dim, self.filters)
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='img_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            # Mask kernel
            self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))
    
            # Calculate padding size to achieve zero-padding
            self.pconv_padding = (
                (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
                (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
            )
    
            # Window size - used for normalization
            self.window_size = self.kernel_size[0] * self.kernel_size[1]
    
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
            self.built = True
    
        def call(self, inputs, mask=None):
            '''
            We will be using the Keras conv2d method, and essentially we have
            to do here is multiply the mask with the input X, before we apply the
            convolutions. For the mask itself, we apply convolutions with all weights
            set to 1.
            Subsequently, we clip mask values to between 0 and 1
            '''
    
            # Both image and mask must be supplied
            if type(inputs) is not list or len(inputs) != 2:
                raise Exception(
                    'PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))
    
            # Padding done explicitly so that padding becomes part of the masked partial convolution
            images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
            masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)
    
            # Apply convolutions to mask
            mask_output = K.conv2d(
                masks, self.kernel_mask,
                strides=self.strides,
                padding='valid',
                data_format=self.data_format,
                dilation_rate=self.dilation_rate
            )
    
            # Apply convolutions to image
            img_output = K.conv2d(
                (images * masks), self.kernel,
                strides=self.strides,
                padding='valid',
                data_format=self.data_format,
                dilation_rate=self.dilation_rate
            )
    
            # Calculate the mask ratio on each pixel in the output mask
            mask_ratio = self.window_size / (mask_output + 1e-8)
    
            # Clip output to be between 0 and 1
            mask_output = K.clip(mask_output, 0, 1)
    
            # Remove ratio values where there are holes
            mask_ratio = mask_ratio * mask_output
    
            # Normalize iamge output
            img_output = img_output * mask_ratio
    
            # Apply bias only to the image (if chosen to do so)
            if self.use_bias:
                img_output = K.bias_add(
                    img_output,
                    self.bias,
                    data_format=self.data_format)
    
            # Apply activations on the image
            if self.activation is not None:
                img_output = self.activation(img_output)
    
            return [img_output, mask_output]
    
        def compute_output_shape(self, input_shape):
            if self.data_format == 'channels_last':
                space = input_shape[0][1:-1]
                new_space = []
                for i in range(len(space)):
                    new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding='same',
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                    new_space.append(new_dim)
                new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
                return [new_shape, new_shape]
            if self.data_format == 'channels_first':
                space = input_shape[2:]
                new_space = []
                for i in range(len(space)):
                    new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding='same',
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                    new_space.append(new_dim)
                new_shape = (input_shape[0], self.filters) + tuple(new_space)
                return [new_shape, new_shape]
    
    
    def get_weight(shape, gain=np.sqrt(2)):
        fan_in = np.prod(shape[:-1])
        std = gain / np.sqrt(fan_in)
        w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
        return w
    
    
    def apply_bias(x):
        b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
        b = tf.cast(b, x.dtype)
        if len(x.shape) == 2:
            return x + b
        return x + tf.reshape(b, [1, -1, 1, 1])
    
    
    def Pconv2d_bias(x, fmaps, kernel, mask_in=None):
        assert kernel >= 1 and kernel % 2 == 1
        x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
        mask_in = tf.pad(mask_in, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT", constant_values=1)
        conv, mask = PConv2D(fmaps, kernel, strides=1, padding='valid',
                             data_format='channels_first')([x, mask_in])
        return conv, mask
    
    
    def conv2d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
        assert kernel >= 1 and kernel % 2 == 1
        w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain)
        w = tf.cast(w, x.dtype)
        x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
        return apply_bias(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW'))
    
    
    def Pmaxpool2d(x, k=2, mask_in=None):
        ksize = [1, 1, k, k]
        x = tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
        mask_out = tf.nn.max_pool(mask_in, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
        return x, mask_out
    
    
    def maxpool2d(x, k=2):
        ksize = [1, 1, k, k]
        return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
    
    
    def upscale2d(x, factor=2):
        assert isinstance(factor, int) and factor >= 1
        if factor == 1: return x
        with tf.variable_scope('Upscale2D'):
            s = x.shape
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
            return x
    
    
    def conv_lr(name, x, fmaps, p=0.7):
        with tf.variable_scope(name):
            x = tf.nn.dropout(x, p)
            return tf.nn.leaky_relu(conv2d_bias(x, fmaps, 3), alpha=0.1)
    
    
    def conv(name, x, fmaps, p):
        with tf.variable_scope(name):
            x = tf.nn.dropout(x, p)
            return tf.nn.sigmoid(conv2d_bias(x, fmaps, 3, gain=1.0))
    
    
    def Pconv_lr(name, x, fmaps, mask_in):
        with tf.variable_scope(name):
            x_out, mask_out = Pconv2d_bias(x, fmaps, 3, mask_in=mask_in)
            return tf.nn.leaky_relu(x_out, alpha=0.1), mask_out
    
    
    def autoencoder(x, mask, channel=3, width=256, height=256, p=0.7, **_kwargs):
        x.set_shape([None, channel, height, width])
        mask.set_shape([None, channel, height, width])
        skips = [x]
    
        n = x
        n, mask = Pconv_lr('enc_conv0', n, 48, mask_in=mask)
        n, mask = Pconv_lr('enc_conv1', n, 48, mask_in=mask)
        n, mask = Pmaxpool2d(n, mask_in=mask)
        skips.append(n)
    
        n, mask = Pconv_lr('enc_conv2', n, 48, mask_in=mask)
        n, mask = Pmaxpool2d(n, mask_in=mask)
        skips.append(n)
    
        n, mask = Pconv_lr('enc_conv3', n, 48, mask_in=mask)
        n, mask = Pmaxpool2d(n, mask_in=mask)
        skips.append(n)
    
        n, mask = Pconv_lr('enc_conv4', n, 48, mask_in=mask)
        n, mask = Pmaxpool2d(n, mask_in=mask)
        skips.append(n)
    
        n, mask = Pconv_lr('enc_conv5', n, 48, mask_in=mask)
        n, mask = Pmaxpool2d(n, mask_in=mask)
        n, mask = Pconv_lr('enc_conv6', n, 48, mask_in=mask)
    
        # -----------------------------------------------
        n = upscale2d(n)
        n = concat(n, skips.pop())
        n = conv_lr('dec_conv5', n, 96, p=p)
        n = conv_lr('dec_conv5b', n, 96, p=p)
        
    
        n = upscale2d(n)
        
        n = concat(n, skips.pop())
        n = conv_lr('dec_conv4', n, 96, p=p)
        n = conv_lr('dec_conv4b', n, 96, p=p)
    
        n = upscale2d(n)
        n = concat(n, skips.pop())
        n = conv_lr('dec_conv3', n, 96, p=p)
        n = conv_lr('dec_conv3b', n, 96, p=p)
    
        n = upscale2d(n)
        n = concat(n, skips.pop())
        n = conv_lr('dec_conv2', n, 96, p=p)
        n = conv_lr('dec_conv2b', n, 96, p=p)
    
        n = upscale2d(n)
        n = concat(n, skips.pop())
        n = conv_lr('dec_conv1a', n, 64, p=p)
        n = conv_lr('dec_conv1b', n, 32, p=p)
        n = conv('dec_conv1', n, channel, p=p)
    
        return n
    
    
    def concat(x, y):
        bs1, c1, h1, w1 = x.shape.as_list()
        bs2, c2, h2, w2 = y.shape.as_list()
        x = tf.image.crop_to_bounding_box(tf.transpose(x, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
        y = tf.image.crop_to_bounding_box(tf.transpose(y, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
        return tf.transpose(tf.concat([x, y], axis=3), [0, 3, 1, 2])
    
    
    def build_denoising_unet(noisy, p=0.7, is_realnoisy=False):
        _, h, w, c = np.shape(noisy)
        noisy_tensor = tf.identity(noisy)
        is_flip_lr = tf.placeholder(tf.int16)
        is_flip_ud = tf.placeholder(tf.int16)
        noisy_tensor = data_arg(noisy_tensor, is_flip_lr, is_flip_ud)
        response = tf.transpose(noisy_tensor, [0, 3, 1, 2])
        mask_tensor = tf.ones_like(response)
        mask_tensor = tf.nn.dropout(mask_tensor, 0.7) * 0.7
        response = tf.multiply(mask_tensor, response)
        slice_avg = tf.get_variable('slice_avg', shape=[_, h, w, c], initializer=tf.initializers.zeros())
        if is_realnoisy:
            response = tf.squeeze(tf.random_poisson(25 * response, [1]) / 25, 0)
        response = autoencoder(response, mask_tensor, channel=c, width=w, height=h, p=p)
        response = tf.transpose(response, [0, 2, 3, 1])
        mask_tensor = tf.transpose(mask_tensor, [0, 2, 3, 1])
        data_loss = mask_loss(response, noisy_tensor, 1. - mask_tensor)
        response = data_arg(response, is_flip_lr, is_flip_ud)
        avg_op = slice_avg.assign(slice_avg * 0.99 + response * 0.01)
        our_image = response
    
        training_error = data_loss
        tf.summary.scalar('data loss', data_loss)
    
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=3)
        model = {
            'training_error': training_error,
            'data_loss': data_loss,
            'saver': saver,
            'summary': merged,
            'our_image': our_image,
            'is_flip_lr': is_flip_lr,
            'is_flip_ud': is_flip_ud,
            'avg_op': avg_op,
            'slice_avg': slice_avg,
        }
    
        return model
    
    
    def mask_loss(x, labels, masks):
        cnt_nonzero = tf.to_float(tf.count_nonzero(masks))
        loss = tf.reduce_sum(tf.multiply(tf.math.pow(x - labels, 2), masks)) / cnt_nonzero
        return loss
    
    
    def data_arg(x, is_flip_lr, is_flip_ud):
        x = tf.cond(is_flip_lr > 0, lambda: tf.image.flip_left_right(x), lambda: x)
        x = tf.cond(is_flip_ud > 0, lambda: tf.image.flip_up_down(x), lambda: x)
        return x
    
    
    
    
    
    def train(file_path, dropout_rate, sigma=25, is_realnoisy=False):
        print(file_path)
        tf.reset_default_graph()
        gt = util.load_np_image(file_path)
        _, w, h, c = np.shape(gt)
        model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
        os.makedirs(model_path, exist_ok=True)
        noisy = util.add_gaussian_noise(gt, model_path, sigma)
        minner = np.amin(noisy)
        noisy = noisy-minner
        maxer = np.amax(noisy)
        noisy = noisy/maxer
        model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)
    
        loss = model['training_error']
        summay = model['summary']
        saver = model['saver']
        our_image = model['our_image']
        is_flip_lr = model['is_flip_lr']
        is_flip_ud = model['is_flip_ud']
        avg_op = model['avg_op']
        slice_avg = model['slice_avg']
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
        avg_loss = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(model_path, sess.graph)
            for step in range(N_STEP):
                feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                               feed_dict=feet_dict)
                avg_loss += loss_value
                if (step + 1) % N_SAVE == 0:
    
                    print("After %d training step(s)" % (step + 1),
                          "loss  is {:.9f}".format(avg_loss / N_SAVE))
                    avg_loss = 0
                    sum = np.float32(np.zeros(our_image.shape.as_list()))
                    for j in range(N_PREDICTION):
                        feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                        o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                        sum += o_image
                    o_image = np.squeeze(np.float32((sum / N_PREDICTION) * maxer + minner))
                    o_avg = np.squeeze(np.float32(o_avg * 255))
                    if is_realnoisy:
                        imwrite(model_path + 'Self2Self-' + str(step + 1) + '.tif', o_avg)
                    else:
                        imwrite(model_path + 'Self2Self-' + str(step + 1) + '.tif', o_image)
                    saver.save(sess, model_path + "model.ckpt-" + str(step + 1))
    
                summary_writer.add_summary(merged, step)
    
    path = './testsets/BSD68/'
    file_list = os.listdir(path)
    sigma = 0
    for file_name in file_list:
        if not os.path.isdir(path + file_name):
            train(path + file_name, 0.2, sigma)


    