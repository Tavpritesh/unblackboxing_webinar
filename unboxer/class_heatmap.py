import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
from keras.preprocessing.image import img_to_array
from keras import activations
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from vis.visualization import visualize_cam, visualize_saliency
from vis.utils import utils

from unboxer.utils import img2tensor, get_pred_text_label, softmax

    
class ClassHeatmap():
    def __init__(self, cam_model, img_shape):
        self.model_ = cam_model             
        self.img_shape_ = img_shape
    
    def generate_cam(self, img, label_id):
        layer_name = 'predictions'
        layer_idx = [idx for idx, layer in enumerate(self.model_.layers) 
                     if layer.name == layer_name][0]

        bgr_img = utils.bgr2rgb(img)
        img_input = np.expand_dims(img_to_array(bgr_img), axis=0)

        heatmap = visualize_cam(self.model_, layer_idx, [label_id], img)

        return heatmap

    def generate_saliency(self, img, label_id):
        layer_name = 'predictions'
        layer_idx = [idx for idx, layer in enumerate(self.model_.layers) 
                     if layer.name == layer_name][0]

        bgr_img = utils.bgr2rgb(img)
        img_input = np.expand_dims(img_to_array(bgr_img), axis=0)

        heatmap = visualize_saliency(self.model_, layer_idx, [label_id], img)

        return heatmap

    def plot_cam(self, img_path):
        return self.plot(self.generate_cam, img_path)
    
    def plot_saliency(self, img_path):
        return self.plot(self.generate_saliency, img_path)
    
    def plot(self, vis_func, img_path):
        img = utils.load_img(img_path, target_size=self.img_shape_)
        img = img[:,:,:3]
        
        predictions = self.model_.predict(img2tensor(img, self.img_shape_))
        predictions = softmax(predictions)
        text_prediction = decode_predictions(predictions)
        
        def _plot(label_id):
            label_id = int(label_id)
            text_label = get_pred_text_label(label_id)
            label_proba = np.round(predictions[0,label_id], 4)
            heatmap = vis_func(img,label_id)
            for p in text_prediction[0]:
                print(p[1:]) 
            plt.title('label:%s\nscore:%s'%(text_label,label_proba))
            plt.imshow(heatmap)
            plt.show()
            
        return interact(_plot, label_id='1')
    
def prep_model_for_cam(model, out_layer_name='predictions'):
    layer_idx = utils.find_layer_idx(model, out_layer_name)

    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    return model