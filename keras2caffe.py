import caffe
from caffe import layers as L
from caffe import params as P
import math
import keras
from keras.models import load_model
from keras.models import model_from_json
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import numpy as np

def load_keras_model(filepath):
    model = load_model(filepath)
    return model

def generate_caffe_model(kModel, filepath, weights_filepath):
    n = caffe.NetSpec()
    layers = kModel.layers
    net_params = dict()

    for layer in layers:
        blobs = layer.get_weights()
        generate_layer(blobs, layer, n, net_params)

    print("Converting model to proto and converting weights")
    make_net(n, filepath)
    # write_caffe_model(n,filepath)
    caffe_model = caffe.Net(filepath, caffe.TEST)
    for layer in caffe_model.params.keys():
        for i in range(0, len(caffe_model.params[layer])):
            print(layer + ": ")
            print(net_params[layer][i].shape,caffe_model.params[layer][i].data.shape)
            # print(dir(caffe_model.params[layer]))
            caffe_model.params[layer][i].data[...] = net_params[layer][i]

    caffe_model.save(weights_filepath)

    return n, caffe_model

def generate_layer(blobs, layer, n, net_params):
    """
    Parameters: blobs: weights for keras, layer: keras layer, n: Caffe NetSpec,
    net_params: Dictionary to store Caffe weights
    """
    if type(layer) == keras.layers.InputLayer:
        name = layer.name
        input_shape=list(layer.batch_input_shape)
        input_shape=[1,input_shape[3],input_shape[1],input_shape[2]]
        n[name] = L.Input(shape=[dict(dim=input_shape)])
        print(f'generate {name} ok...')
        
    elif type(layer) == keras.layers.Dense:
        name = layer.name
        config = layer.get_config()
        use_bias = config['use_bias']
        if use_bias == None:
            use_bias = False
        if use_bias:
            net_params[name] = (np.array(blobs[0]).transpose(1, 0), np.array(blobs[1]))
        else:
            net_params[name] = (blobs[0])
        in_nodes = get_input_nodes(layer)
        n[name] = L.InnerProduct(n[in_nodes[0].name], num_output=layer.units,bias_term=use_bias)
        if layer.activation is not None and layer.activation.__name__ != 'linear':
            name_act = name + "_activation_" + layer.activation.__name__  # get function string
            n[name_act] = apply_activation(layer, n[name])
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.Flatten:
        raise Exception(f"{layer.name} is not implemented")

    elif type(layer) == keras.layers.Dropout:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        n[name] = L.Dropout(n[in_nodes[0].name], dropout_ratio=layer.rate, in_place=True)
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.Add:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        network_layers = []
        for ref in in_nodes:
            network_layers.append(n[ref.name])
        n[name] = L.Eltwise(*network_layers, operation=1)  # 1 is SUM
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.Multiply:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        network_layers = []
        for ref in in_nodes:
            network_layers.append(n[ref.name])
        n[name] = L.Eltwise(*network_layers, operation=0)
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.Concatenate:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        network_layers = []
        for ref in in_nodes:
            network_layers.append(n[ref.name])
        n[name] = L.Concat(*network_layers, axis=1)
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.Maximum:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        network_layers = []
        for ref in in_nodes:
            network_layers += n[ref.name]
        n[name] = L.Eltwise(*network_layers, operation=2)
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.Conv2DTranspose:
        '''
            DeconvolutionLayer：
            output = (input - 1) * stride + kernel_size - 2 * pad;
            kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
            num_output: {{C}} group: {{C}}
            pad: {{ceil((factor - 1) / 2.)}}
        '''
        name = layer.name
        in_nodes = get_input_nodes(layer)
        # Stride
        if layer.strides is None:
            stride = (1, 1)
        else:
            stride = layer.strides

        # if layer.padding == 'same':  # Calculate the padding for 'same'
        #     padding = [layer.kernel_size[0] // 2, layer.kernel_size[1] // 2]
        # else:
        #     padding = [0, 0]  # If padding is valid(aka no padding)

        config = layer.get_config()
        use_bias = config['use_bias']
        if use_bias==None:
            use_bias=False
        n[name] = L.Deconvolution(n[in_nodes[0].name],convolution_param=dict(kernel_h=layer.kernel_size[0],kernel_w=layer.kernel_size[1],
                stride_h =stride[0],stride_w =stride[1],num_output=layer.filters, pad_h =math.ceil((stride[0] - 1) / 2.),
                                                        pad_w =math.ceil((stride[1] - 1) / 2.),bias_term=use_bias))
        blobs[0] = np.array(blobs[0]).transpose(3, 2, 0, 1)
        net_params[name] = blobs
        if layer.activation is not None and layer.activation.__name__ != 'linear':
            name_act = name + "_activation_" + layer.activation.__name__  # get function string
            n[name_act] = apply_activation(layer, n[name])

        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.BatchNormalization:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        n[name] = L.BatchNorm(n[in_nodes[0].name], moving_average_fraction=layer.momentum, eps=layer.epsilon,use_global_stats=True)
        variance = np.array(blobs[-1])
        mean = np.array(blobs[-2])

        config = layer.get_config()
        param = dict()
        if config['scale']:
            gamma = np.array(blobs[0])
        else:
            gamma = np.ones(mean.shape, dtype=np.float32)

        if config['center']:
            beta = np.array(blobs[1])
            param['bias_term'] = True
        else:
            beta = np.zeros(mean.shape, dtype=np.float32)
            param['bias_term'] = False

        net_params[name] = (mean, variance, np.array([1.0]))

        # Scale after batchNorm
        name_scale = name + '_scale'
        n[name_scale] = L.Scale(n[name], in_place=True, scale_param=param)
        net_params[name_scale] = (gamma, beta)
        print(f'generate {name} ok...')

    # TODO Needs to be implemented
    elif type(layer) == keras.layers.Conv1D:
        raise Exception(f"{layer.name} is not implemented")

    elif type(layer) == keras.layers.ZeroPadding2D:
        print(f"{layer.name} is passed...")

    elif type(layer) == keras.layers.Conv2D:
        '''
           ConvolutionLayer：
           output = (input + 2 * pad  - kernel_size)  / stride + 1;
           kernel_shape: [out,in,k_size_h,k_size_w]
        '''
        name = layer.name

        # Padding
        if layer.padding == 'same':  # Calculate the padding for 'same'
            padding = [layer.kernel_size[0] // 2, layer.kernel_size[1] // 2]
        else:
            padding = [0, 0]  # If padding is valid(aka no padding)

        in_nodes = get_input_nodes(layer)
        if type(in_nodes[0])==keras.layers.ZeroPadding2D:
            in_nodes = get_input_nodes(in_nodes[0])
            padding = [layer.kernel_size[0] // 2, layer.kernel_size[1] // 2]

        if layer.strides is None:
            stride = (1, 1)
        else:
            stride = layer.strides

        # TODO The rest of the arguements including bias, regulizers, dilation,
        config = layer.get_config()
        # print(config)
        # get bias parameter
        use_bias = config['use_bias']
        if use_bias==None:
            use_bias=False
        n[name] = L.Convolution(n[in_nodes[0].name], kernel_h=layer.kernel_size[0],
                                kernel_w=layer.kernel_size[1], stride_h=stride[0],
                                stride_w=stride[1], num_output=layer.filters, pad_h=padding[0], pad_w=padding[1],
                                bias_term=use_bias)
        # weights = blobs
        blobs[0] = np.array(blobs[0]).transpose((3, 2, 0, 1))
        # print(blobs[0].shape)
        net_params[name] = blobs
        if layer.activation is not None and layer.activation.__name__ != 'linear':
            name_act = name + "_activation_" + layer.activation.__name__  # get function string
            n[name_act] = apply_activation(layer, n[name])

        print(f'generate {name} ok...')


    elif type(layer) == keras.layers.MaxPooling2D or type(layer) == keras.layers.AveragePooling2D:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        if type(layer) == keras.layers.MaxPooling2D:
            pool = P.Pooling.MAX
        else:  # NOTE AveragePooling needs to be implemented
            pool = P.Pooling.AVE
        # Padding
        # TODO The rest of the arguements including bias, regulizers, dilatin,
        if layer.strides is None:
            stride = (1, 1)
        else:
            stride = layer.strides
        # Padding
        if layer.padding == 'same':  # Calculate the padding for 'same'
            padding = [layer.pool_size[0] // 2, layer.pool_size[1] // 2]
        else:
            padding = [0, 0]  # If padding is valid(aka no padding)
        n[name] = L.Pooling(n[in_nodes[0].name], kernel_h=layer.pool_size[0],
                            kernel_w=layer.pool_size[1], stride_h=stride[0],
                            stride_w=stride[1], pad_h=padding[0], pad_w=padding[1],
                            pool=pool)
        print(f'generate {name} ok...')

    # Activation (wrapper for activations) and Advanced Activation Layers
    elif type(layer) == keras.layers.Activation:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        n[name] = apply_activation(layer, n[in_nodes[0].name])  # TODO: Assert only 1
        print(f'generate {name} ok...')

    # Caffe lacks intializer, regulizer, and constraint params
    elif type(layer) == keras.layers.LeakyReLU:
        # TODO: figure out how to pass Leaky params
        name = layer.name
        in_nodes = get_input_nodes(layer)
        n[name] = L.PReLU(n[in_nodes[0].name])
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.PReLU:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        n[name] = L.PReLU(n[in_nodes[0].name])
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.ELU:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        n[name] = L.ELU(n[in_nodes[0].name], layer.alpha)
        print(f'generate {name} ok...')

    elif type(layer) == keras.layers.GlobalAveragePooling2D:
        name = layer.name
        in_nodes = get_input_nodes(layer)
        n[name] = L.Pooling(n[in_nodes[0].name], kernel_size=layer.kernel_size[0], stride=layer.strides[0], pad=layer.kernel_size[0]//2, pool=P.Pooling.AVE)
        print(f'generate {name} ok...')

    else:
        raise Exception("Cannot convert model." + layer.name + " is not supported.")

def get_input_nodes(layer):
    in_nodes = []
    for node in layer._inbound_nodes:  # get inbound nodes to current layer
        node_list = node.inbound_layers  # get layers pointing to this node
        in_nodes = in_nodes + node_list
    return in_nodes

def apply_activation(layer, bottom):
    if keras.activations.serialize(layer.activation) == 'relu':
        return L.ReLU(bottom, in_place=True)
    elif keras.activations.serialize(layer.activation) == 'softmax':
        return L.Softmax(bottom)  # Cannot extract axis from model, so default to -1
    elif keras.activations.serialize(layer.activation) == 'softsign':
        # Needs to be implemented in caffe2dml
        raise Exception("softsign is not implemented")
    elif keras.activations.serialize(layer.activation) == 'elu':
        return L.ELU(bottom)
    elif keras.activations.serialize(layer.activation) == 'selu':
        # Needs to be implemented in caffe2dml
        raise Exception("SELU activation is not implemented")
    elif keras.activations.serialize(layer.activation) == 'sigmoid':
        return L.Sigmoid(bottom)
    elif keras.activations.serialize(layer.activation) == 'tanh':
        return L.TanH(bottom)

def make_net(n,prototxt):
    with open(prototxt, 'w') as fpb:
        print(n.to_proto(), file=fpb)

if __name__=='__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    kModel=load_keras_model('../your_keras_model.h5')
    prototxt='../your_caffe_model.prototxt'
    caffemodel='../your_caffe_model.caffemodel'
    generate_caffe_model(kModel, prototxt, caffemodel)
