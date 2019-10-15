import numpy as np
import tensorflow.keras as keras
from scipy.ndimage import zoom
import os
import nibabel as nib

class FMRIDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, dataset_dir, batch_size):
        'Initialization'
        self.x_dim = 49
        self.y_dim = 58
        self.z_dim = 47
        self.time_length = 177
        self.img_dim = (self.x_dim, self.y_dim, self.z_dim) 
        self.dim = [self.time_length, 28, 28, 28, 1] # [time, x, y, z, c]
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = 1
        self.n_classes = 1
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
                
        for i, img_path in enumerate(list_IDs_temp):
            X[i,] = self.preprocess_image(img_path)
            y[i] = self.labels[img_path]

        return X,y

    # Image Preprocessing Methods
    def preprocess_image(self, img_path):
        img = nib.load(os.path.join(self.dataset_dir, img_path))

        pp_img = None
        if img.shape[3] > self.time_length:
            pp_img = self.truncate_image(img)
        elif img.shape[3] < self.time_length:
            pp_img = self.pad_image(img)
        else:
            pp_img = img.get_fdata()

        # For each image at the index-th time step, do this
        new_x=28/49
        new_y=28/58
        new_z=28/47
        
        new_img = []
        for index in range(self.time_length):
            z_img = zoom(pp_img[:,:,:,index], (new_x,new_y,new_z), order=1)
            new_img.append(z_img.reshape((28,28,28,1)))
        
        f_img = np.array(new_img)
        return f_img
    
    def truncate_image(self, img):
        return img.get_fdata()[:,:,:,:self.time_length]

    def pad_image(self, img):
        img_padding = np.expand_dims(np.zeros((self.x_dim,self.y_dim,self.z_dim)), axis=3)
        amt_to_fill = self.time_length - img.get_fdata().shape[3]
        padded_img = img.get_fdata()
        for _ in range(amt_to_fill):
            padded_img = np.append(arr=padded_img, values=img_padding, axis=3)

        return padded_img