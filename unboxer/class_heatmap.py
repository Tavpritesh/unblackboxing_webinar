import numpy as np

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input,decode_predictions

import matplotlib.pyplot as plt
from ipywidgets import interact


from unboxer.vis_py3.visualization import visualize_cam, visualize_saliency
from unboxer.vis_py3.utils.vggnet import VGG16
from unboxer.vis_py3.utils.utils import load_img, bgr2rgb

from unboxer.utils import img2tensor, get_pred_text_label, softmax

    
class ClassHeatmap():
    def __init__(self, model_architecture, img_shape):
        if model_architecture == 'vgg16':
            self.model_ = VGG16(weights='imagenet', include_top=True)
        else:
            raise NotImplementedError
            
        self.img_shape_ = img_shape
    
    def generate_cam(self, img, label_id):
        layer_name = 'predictions'
        layer_idx = [idx for idx, layer in enumerate(self.model_.layers) 
                     if layer.name == layer_name][0]

        bgr_img = bgr2rgb(img)
        img_input = np.expand_dims(img_to_array(bgr_img), axis=0)

        heatmap = visualize_cam(self.model_, layer_idx, [label_id], img)

        return heatmap

    def generate_saliency(self, img, label_id):
        layer_name = 'predictions'
        layer_idx = [idx for idx, layer in enumerate(self.model_.layers) 
                     if layer.name == layer_name][0]

        bgr_img = bgr2rgb(img)
        img_input = np.expand_dims(img_to_array(bgr_img), axis=0)

        heatmap = visualize_saliency(self.model_, layer_idx, [label_id], img)

        return heatmap

    def plot_cam(self, img_path):
        return self.plot(self.generate_cam, img_path)
    
    def plot_saliency(self, img_path):
        return self.plot(self.generate_saliency, img_path)
    
    def plot(self, vis_func, img_path):
        img = load_img(img_path, target_size=self.img_shape_)
        img = img[:,:,:3]
        
        predictions = self.model_.predict(img2tensor(img, self.img_shape_))
        predictions = softmax(predictions)
        text_prediction = decode_predictions(predictions)
        
        def _plot(label_id):
            label_id = int(label_id)
            text_label = get_pred_text_label(label_id)
            heatmap = vis_func(img,label_id)
            for p in text_prediction[0]:
                print(p[1:]) 
            plt.title(text_label)
            plt.imshow(heatmap)
            plt.show()
            
        return interact(_plot, label_id='1')