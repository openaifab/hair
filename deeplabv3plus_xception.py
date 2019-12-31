import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
#from scipy.misc import imread
#import imageio


class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513 #圖片長寬
    FROZEN_GRAPH_NAME = 'frozen' #_inference_graph
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()  
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)
    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap
def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]

def run_deeplabv3plus_xception(model_input,photo_input,output_file,show=False):
    MODEL = model_input
    original_im = Image.open(photo_input)
    width, height = original_im.size
    resized_im, seg_map = MODEL.run(original_im)
    cm = seg_map
    img = np.array(resized_im)
    rows = cm.shape[0]
    cols = cm.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if cm[x][y] == 0:
                img[x][y] = np.array([255, 255, 255], dtype='uint8')
    img = Image.fromarray(img)
    img_convert = img.resize((width, height),Image.ANTIALIAS)
    img_convert.save(output_file)
    if(width>513 or height>513):
        print("deeplabv3+ xception/mobilenet 成功：但圖片被壓縮")
    else:
        print("deeplabv3+ xception/mobilenet 成功")
    if show:
        return(img_convert)
 
