import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from sklearn.datasets import load_files

# InceptionV3 requires input shapes (of images) to be (299x299).
INPUT_SHAPE = (299, 299)


# Load the dataset.
def load_dataset(data_path, shuffle=None, p=1):
    kwargs = {}
    if shuffle is not None:
        kwargs["shuffle"] = shuffle
    data = load_files(data_path, **kwargs)
    img_files = np.array(data["filenames"])
    targets = np_utils.to_categorical(np.array(data["target"]), 3)
    length = int(float(p) * len(targets))
    return img_files[:length], targets[:length]


# Load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network.
def path_to_tensor(img_path):
    # Loads RGB image as PIL.Image.Image type.
    image = load_img(img_path, target_size=INPUT_SHAPE)
    # Convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3).
    image = img_to_array(image)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    image = np.expand_dims(image, axis=0)
    return image


def paths_to_tensor(image_paths):
    return np.vstack([path_to_tensor(path) for path in image_paths])
