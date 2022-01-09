from glob import glob

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from tqdm import tqdm

from utils import load_dataset, paths_to_tensor

# Load the data for training.
train_files, train_targets = load_dataset("data/train")
valid_files, valid_targets = load_dataset("data/valid")

# Load lables.
label_name = [item[11:-1] for item in sorted(glob("data/train/*/"))]

# Summary of the dataset.
print("Train Files Size: {}".format(len(train_files)))
print("Train Files Shape: {}".format(train_files.shape))
print("Target Shape: {}".format(train_targets.shape))
print("Label Names: {}".format(label_name))

""" Transfer learning using Inception V3 """
# Load the Inception V3 model as well as the network weights from disk.
print("\n[INFO] loading CNN Model\n")
transfer_model = InceptionV3(include_top=False, weights="imagenet")

# Creating 4D tensors from the images in the dataset.
print("\n[INFO] Loading and Pre-processing images...")
train_tensors = paths_to_tensor(tqdm(train_files))
valid_tensors = paths_to_tensor(tqdm(valid_files))

# Preprocess the input for the Inception V3 model.
print("[INFO] This may take some time...")
train_images = preprocess_input(train_tensors)
valid_images = preprocess_input(valid_tensors)

# Getting data ready by processing it from the Inception V3 model & extracting the important features.
train_data = transfer_model.predict(train_images)
valid_data = transfer_model.predict(valid_images)

""" We are totally ready now to train our network """
print("[INFO] Train data shape: {}".format(train_data.shape))

# Model Architecture.
CNN_model = Sequential()
CNN_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(1024, activation="relu"))
CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(3, activation="softmax"))

# Model Summary.
CNN_model.summary()

# Compile the model.
CNN_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define the callbacks.
checkpointer = ModelCheckpoint(
    filepath="saved_models_weights_checkpointer/weights.best.model.hdf5", verbose=1, save_best_only=True
)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="min")

# Train the model.
CNN_model.fit(
    train_data,
    train_targets,
    validation_data=(valid_data, valid_targets),
    epochs=60,
    batch_size=200,
    callbacks=[checkpointer, early_stopping],
    verbose=1,
)

""" Save the model """
# Serialize model to JSON.
model_json = CNN_model.to_json()
with open("models/CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5.
CNN_model.save_weights("weights/CNN_model.h5")
print("\nSaved model to disk.\n")
