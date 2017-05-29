import sys, os

import numpy as np

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input,decode_predictions

if sys.version_info.major == 3:
    from vis_py3.visualization import visualize_activation
    from vis_py3.utils.vggnet import VGG16
    from vis_py3.utils.utils import load_img, bgr2rgb
else:
    from vis.visualization import visualize_activation
    from vis.utils.vggnet import VGG16
    from vis.utils.utils import load_img, bgr2rgb   
    
import matplotlib.pyplot as plt
from ipywidgets import interact
from IPython.html import widgets
from IPython.display import display


class DeepVis():
    def __init__(self, model_architecture, save_dir):
        if model_architecture == 'vgg16':
            self.model_ = VGG16(weights='imagenet', include_top=True)
            self.layer_filter_ids_ = self._build_layer_filter_dict(self.model_)  
        else:
            pass
         
        self.save_dir_ = save_dir
    
    def browse(self):
        def plot(layer_id, filter_id):
            filepath = '{}/{}/{}/img.jpg'.format(self.save_dir_, 
                                                 layer_id, filter_id)
            img = plt.imread(filepath)
            plt.imshow(img)
            plt.show()
        return interact(plot, layer_id='1',filter_id='0')
        
    def generate_max_activation_images(self, layer_ids):
    
        for layer_id in layer_ids:
            for filter_id in range(self.layer_filter_ids_[layer_id]):
                print('layer:{} filter:{}'.format(layer_id,filter_id))
                maximal_activation_image = self.find_mai(layer_id,filter_id)
                self.save(layer_id, filter_id, maximal_activation_image)
                
    def find_mai(self, layer_id, filter_id):
        no_tv_seed_img = visualize_activation(self.model_, layer_id, 
                                              filter_indices=[filter_id],
                                              tv_weight=0, verbose=False)
        post_tv_img = visualize_activation(self.model_, layer_id, 
                                           filter_indices=[filter_id],
                                           tv_weight=1, seed_img=no_tv_seed_img, 
                                           verbose=False, max_iter=100)
        return post_tv_img
    
    def save(self, layer_id, filter_id, img):
        directory = '{}/{}/{}'.format(self.save_dir_, layer_id, filter_id)
        
        if not os.path.exists(directory): os.makedirs(directory)
        filepath = os.path.join(directory,'img.jpg')
        plt.imsave(filepath, img)
         
    def _build_layer_filter_dict(self, model):
        layer_filter_dict = {}
        for i,l in enumerate(model.layers):
            try:
                filter_shape = l.get_weights()[0].shape
            except Exception:
                continue
            layer_filter_dict[i] = filter_shape[-1]
        return layer_filter_dict
    
if __name__ == '__main__':
    
    deep_vis = DeepVis(model_architecture='vgg16', save_dir='/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/data/filter_images')
    conv_layer_ids = [11, 12, 13, 15, 16, 17]
    deep_vis.generate_max_activation_images(conv_layer_ids)