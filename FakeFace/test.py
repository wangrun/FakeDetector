import random
import numpy as np
from collections import defaultdict

from Model1 import Model1
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model,load_model

def init():

    model = load_model('./Model1.h5')
    # model.save_weights('./Model1.h5')
    # model.load_weights('./Model1.h5')
    model.load_weights('./Model1.h5')

    # (_, _), (x_test, _) = mnist.load_data()
    # img_rows, img_cols = 28, 28
    # input_shape = (img_rows, img_cols, 1)
    # input_tensor = Input(shape=input_shape)
    # model = Model1(input_tensor=input_tensor)
    # model.load_weights('./Model1.h5')


    model_layer_dict=init_coverage_tables(model)
    gen_img=generate_images()
    threshold=0.6
    update_coverage(gen_img,model,model_layer_dict,threshold)

def generate_images():
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    (_, _), (x_test, _) = mnist.load_data()

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255

    # gen_img=np.expand_dims(random.choice(x_test),axis=0)
    gen_img=np.expand_dims(x_test[0],axis=0)

    return gen_img

def init_coverage_tables(model):
    model_layer_dict = defaultdict(bool)
    init_dict(model, model_layer_dict)
    return model_layer_dict

def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False

def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    # print len(layer_names)
    print layer_names


    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])

    print len(intermediate_layer_model.layers)
    # print intermediate_layer_model.summary()

    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for n in range(0,len(intermediate_layer_outputs)):
        print type(intermediate_layer_outputs[n]),intermediate_layer_outputs[n].shape,intermediate_layer_outputs[n].size

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        print scaled.shape[-1]
        for num_neuron in xrange(scaled.shape[-1]):
            print ":::::::::::::::::",np.mean(scaled[..., num_neuron])
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True

    print neuron_covered(model_layer_dict)

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def main():
    init()


if __name__ == '__main__':
    main()
