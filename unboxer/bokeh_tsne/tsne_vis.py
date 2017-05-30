import pandas as pd

from sklearn.manifold import TSNE

from unboxer.bokeh_tsne.utils import *
from unboxer.bokeh_tsne.hover_scatter import scatterplot


class TsneVis():
    def __init__(self, model, feature_layer_name, **tsne_kwargs):
        self.tsne_model_ = TSNE(**tsne_kwargs)
        self.model_ = model
        self.feature_layer_name_ = feature_layer_name
        
        self.tsne_features_ = None
    
    def plot(self, **kwargs):
        scatterplot(self.tsne_features_, **kwargs)
    
    def fit(self, img_folder, label_df=pd.DataFrame(), batch_size=2):
        img_input_shape = self.model_.input_shape[1:-1]
        img_paths, img_tensor = folder2tensor(img_folder, paths=True, shape=None)
        img_features = self._extract_features(img_tensor, batch_size)
        
        tsne_features = self.tsne_model_.fit_transform(img_features)
        
        df = pd.DataFrame(tsne_features, columns=['x','y'])
        df['img_filepath'] = img_paths
        df.sort_values('img_filepath', inplace=True)

        if label_df.empty:
            df['label'] = 0
        else:
            label_df.sort_values('img_filepath', inplace=True)
            df.reset_index(inplace=True)
            label_df.reset_index(inplace=True)
            df['img_filepath_wtf'] = label_df['img_filepath']
            df['label'] = label_df['label']  
                      
        self.tsne_features_ = df
        
    def _extract_features(self,X, batch_size):
        layer_id = [i for i, l in enumerate(self.model_.layers) 
                    if l.name == self.feature_layer_name_][0]
        img_features = get_layer_output(self.model_, 
                                        layer=layer_id, X=X, batch_size=batch_size)

        new_shape = np.product(img_features.shape[1:])
        nr_imgs = img_features.shape[0]
        img_features = np.reshape(img_features,(nr_imgs, new_shape))
        return img_features