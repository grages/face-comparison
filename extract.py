import caffe
import numpy as np
import cv2

def initvgg():
    print("Begin to import caffemodel")

    model_def = './data/deploy.prototxt'
    model_weights = './data/vgg_face.caffemodel'

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mu = np.array([129.1863, 104.7624, 93.5940])

    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    net.blobs['data'].reshape(1, 3, 224, 224)
    print("Caffemodel online")
    return net,transformer

def face(ima,net,transformer):
    image=cv2.resize(ima,(224,224))
    transformed_image = transformer.preprocess('data', image)

    net.blobs['data'].data[0] = transformed_image
    output = net.forward()

    output_fc = output['fc7'][0]
    return output_fc