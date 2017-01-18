from __future__ import print_function

import os
import time

import numpy as np

from matplotlib import pylab as plt
from PIL import Image

from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from keras import backend as K
from keras.layers import Input

from deepsense import neptune

saved_settings = {
    'cool_trip': {'features': {'block5_conv3': 0.03},
                 'continuity': 0.1,
                 'dream_l2': 0.02,
                 'jitter': 5},
    'hardcore': {'features': {'block5_conv3': 0.9,
                             'block5_conv2': 0.9},
                 'continuity': 0.9,
                 'dream_l2': 0.9,
                 'jitter': 5},
    'bad_trip': {'features': {'block4_conv1': 0.05,
                              'block4_conv2': 0.01,
                              'block4_conv3': 0.01},
                 'continuity': 0.1,
                 'dream_l2': 0.8,
                 'jitter': 5},
    'dreamy': {'features': {'block5_conv1': 0.05,
                            'block5_conv2': 0.02},
               'continuity': 0.1,
               'dream_l2': 0.02,
               'jitter': 0},
}

settings = saved_settings['cool_trip']
continuity = settings['continuity']
dream_l2 = settings['dream_l2']
jitter = settings['jitter']

def set_continuity(value):
    global continuity
    continuity  = float(value)
    return str(continuity)

def set_dream_l2(value):
    global dream_l2
    dream_l2  = float(value)
    return str(dream_l2)
    
def set_jitter(value):
    global jitter
    jitter  = float(value)
    return str(jitter)
    
ctx = neptune.Context()

logging_channel = ctx.job.create_channel(
    name='logging_channel',
    channel_type=neptune.ChannelType.TEXT)

loss_channel = ctx.job.create_channel(
    name='training loss',
    channel_type=neptune.ChannelType.NUMERIC)

result_channel = ctx.job.create_channel(
    name='result image',
    channel_type=neptune.ChannelType.IMAGE)

ctx.job.register_action(name='set continuity', handler = set_continuity)
ctx.job.register_action(name='set dream l2', handler = set_dream_l2)
ctx.job.register_action(name='set jitter', handler = set_jitter)

ctx.job.finalize_preparation()

logging_channel.send(x = time.time(),y = settings)

BASE_IMAGE_PATH = (ctx.params.base_file)
DREAM_ITER = (ctx.params.dream_iter)
OUTPUT_PATH = (ctx.params.output_folder)


# dimensions of the generated picture.
img_width = 300
img_height = 300

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.reshape((img_width, img_height, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def neptune_image(raw_image):
    stylish_image = Image.fromarray(raw_image)
    return neptune.Image(
        name="neptune dreams",
        description="deep dream image",
        data=stylish_image)

if __name__ == "__main__":
    
    
    img_size = (img_width, img_height, 3)
    # this will contain our generated image
    dream = Input(batch_shape=(1,) + img_size)

    logging_channel.send(x = time.time(),y = " Reading model VGG16...")
    # build the VGG16 network with our placeholder
    # the model will be loaded with pre-trained ImageNet weights
    model = vgg16.VGG16(input_tensor=dream,
                        weights='imagenet', include_top=False)
    
    logging_channel.send(x = time.time(),y = "Building Objects...")
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # continuity loss util function
    def continuity_loss(x):
        assert K.ndim(x) == 4
        a = K.square(x[:, :img_width - 1, :img_height-1, :] -
                     x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height-1, :] -
                     x[:, :img_width - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    # define the loss
    loss = K.variable(0.)
    for layer_name in settings['features']:
        # add the L2 norm of the features of a layer to the loss
        assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
        coeff = settings['features'][layer_name]
        x = layer_dict[layer_name].output
        shape = layer_dict[layer_name].output_shape

        # we avoid border artifacts by only involving non-border pixels in the loss
        loss -= coeff * K.sum(K.square(x[:, 2: shape[1] - 2, 2: shape[2] - 2, :])) / np.prod(shape[1:])

        # add continuity loss (gives image local coherence, can result in an artful blur)
        loss += continuity* continuity_loss(dream) / np.prod(img_size)
        # add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
        loss += dream_l2 * K.sum(K.square(dream)) / np.prod(img_size)

        # feel free to further modify the loss as you see fit, to achieve new effects...

        # compute the gradients of the dream wrt the loss
        grads = K.gradients(loss, dream)

        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)

        f_outputs = K.function([dream], outputs)
        def eval_loss_and_grads(x):
            x = x.reshape((1,) + img_size)
            outs = f_outputs([x])
            loss_value = outs[0]
            if len(outs[1:]) == 1:
                grad_values = outs[1].flatten().astype('float64')
            else:
                grad_values = np.array(outs[1:]).flatten().astype('float64')
            return loss_value, grad_values

        # this Evaluator class makes it possible
        # to compute loss and gradients in one pass
        # while retrieving them via two separate functions,
        # "loss" and "grads". This is done because scipy.optimize
        # requires separate functions for loss and gradients,
        # but computing them separately would be inefficient.
        class Evaluator(object):
            def __init__(self):
                self.loss_value = None
                self.grad_values = None

            def loss(self, x):
                assert self.loss_value is None
                loss_value, grad_values = eval_loss_and_grads(x)
                self.loss_value = loss_value
                self.grad_values = grad_values
                return self.loss_value

            def grads(self, x):
                assert self.loss_value is not None
                grad_values = np.copy(self.grad_values)
                self.loss_value = None
                self.grad_values = None
                return grad_values

        logging_channel.send(x = time.time(),y = "Resuming Deep Dream")    

        evaluator = Evaluator()

        # run scipy-based optimization (L-BFGS) over the pixels of the generated image
        # so as to minimize the loss
        x = preprocess_image(BASE_IMAGE_PATH)

        for i in range(DREAM_ITER):
            logging_channel.send(x = time.time(),y = 'Iteration: %s'%i)
            start_time = time.time()

            # add a random jitter to the initial image. This will be reverted at decoding time
            random_jitter = (jitter* 2) * (np.random.random(img_size) - 0.5)
            x += random_jitter

            # run L-BFGS for 7 steps
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                             fprime=evaluator.grads, maxfun=7)
            logging_channel.send(x = time.time(),y = 'Current loss value: %s'%min_val)
            loss_channel.send(x = i,y = float(min_val))

            # decode the dream and save it
            x = x.reshape(img_size)
            x -= random_jitter
            img = deprocess_image(np.copy(x))
            plt.imsave(os.path.join(OUTPUT_PATH,"deep_dream_%s.jpg"%i),img)
            result_channel.send(x = time.time(),y = neptune_image(img))
            end_time = time.time()

            logging_channel.send(x = time.time(),y = 'Iteration %d completed in %ds' % (i, end_time - start_time))  
