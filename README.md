# tensorflow_keras_to_caffe

### This script implements the tensorflow1.x and keras model into a caffe inference model.[zhihu](https://zhuanlan.zhihu.com/p/115081459)

### envs and tools

```python
tensorflow1.13.2、keras2.2.4、caffe1.0、python3.6、pycharm、Anoconda、MMdnn0.2.4、ubuntu18.04
```

* If caffe is only needed for model transformation, you can simply install with conda, command line:

  ```python
  conda install caffe-gpu
  ```

* If your model is now a keras model, then the [MMdnn]( https://github.com/microsoft/MMdnn) is not needed.

* If your model is now a tensorflow model, such as the ckpt model, then you need Microsoft's MMdnn for conversion. Although this tool currently can not seamlessly convert all different frameworks, the conversion between tf and keras is painless. Command line conversion example:

  ```python
  mmconvert -sf tensorflow -in your_model_name.ckpt.meta -iw your_model_name.ckpt --inNodeName input_name --inputShape 224,224,3 --dstNodeName output1 ouput2 -df keras -om your_keras_model.h5
  ```

### Supported layers

```python
- InputLayer
- Dense
- Dropout
- ZeroPadding2D
- Multiply、Concatenate、Maximum、Add
- Conv2D、Conv2DTranspose
- BatchNormalization
- MaxPooling2D、AveragePooling2D、GlobalAveragePooling2D
- relu、prelu、elu、softmax、sigmoid、tanh
```

You can modify the `generate_layer(blobs, layer, n, net_params)`  function to implement your own layer transformation.

### How to use

Specify your keras model path and the name of the caffe model you want, then just run is. like this:

```python
if __name__=='__main__':
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    kModel=load_keras_model('../your_keras_model.h5')
    prototxt='../your_caffe_model.prototxt'
    caffemodel='../your_caffe_model.caffemodel'
    generate_caffe_model(kModel, prototxt, caffemodel)
```

### Tips

* If your model is a tensorflow model and used `conv2d_transpose ` layer (`Deconvolution `in caffe), then you must avoid using high-level api, such as `slim.conv2d_transpose`, and you need to use  `tf.nn.conv2d_transpose` interface. Otherwise, the conversion will fail (it seems that the implementation is different).

* In addition, the padding method of convolution in tensorflow or keras is different from the caffe. When 'same' padding in tf / keras, there is a case only pad the bottom right, but in caffe will pad top, bottom, left and right. That will lead to inaccurate results. To solve this problem, it is recommended to manually add a pad in tensorflow, like this:

  ```python
  def conv2d(inputs, filters, kernel_size,      strides=1,rate=1,biases_initializer=tf.zeros_initializer,  activation_fn=tf.nn.relu):#stride>1时padding，valid卷积实现same
      def _fixed_padding(inputs, kernel_size):
          pad_total = kernel_size - 1
          pad_beg = pad_total // 2
          pad_end = pad_total - pad_beg
  
          padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                          [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
          return padded_inputs
  
      if strides > 1:
          inputs = _fixed_padding(inputs, kernel_size)
      inputs = slim.conv2d(inputs, filters, kernel_size,      stride=strides,rate=rate,biases_initializer=biases_initializer,
                         activation_fn=activation_fn,padding=('SAME' if strides == 1  else 'VALID'),weights_initializer=tf.initializers.he_normal())
      return inputs
  ```

  In this way, removing the pad layer during the conversion process, such as removing the `ZeroPadding2D` layer in keras, can directly use only the convolution layer in caffe to achieve same output.

#### reference

https://github.com/anoojpatel/keras2caffe, I fixed some bugs and simplified the code, and only converted to the inference model.
