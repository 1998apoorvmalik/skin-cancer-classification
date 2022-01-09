import argparse
import csv
from glob import glob

import numpy as np
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import load_dataset, paths_to_tensor

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testSize", default=1, help="Fraction of the test images to use for model evaluation.")
args = vars(ap.parse_args())

""" Transfer learning using Inception V3 """
# Load the Inception V3 model as well as the network weights from disk.
print("[INFO] loading CNN Model")
transfer_model = InceptionV3(include_top=False, weights="imagenet")

# Load the test dataset & preprocess it.
test_files, test_targets = load_dataset("data/test", shuffle=True, p=args["testSize"])
print("\n[INFO] Loading and Pre-processing images...")
test_tensors = paths_to_tensor(tqdm(test_files))
print("[INFO] This may take some time...")
test_images = preprocess_input(test_tensors)
test_data = transfer_model.predict(test_images)
# Load lables.
label_name = [item[11:-1] for item in sorted(glob("data/train/*/"))]
print("[INFO] Label names are: {}".format(label_name))

""" Retrieve the saved CNN model """
# Load json and create model.
json_file = open("models/CNN_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into the new model.
loaded_model.load_weights("weights/CNN_model.h5")
CNN_model = loaded_model
print("[INFO] Loaded model from the disk.")

# Evaluate the model.
predictions = [CNN_model.predict(np.expand_dims(feature, axis=0)) for feature in test_data]
y_true = [np.argmax(i) for i in test_targets]
y_pred = [np.argmax(i[0]) for i in predictions]

# Calculate the classification accuracy.
test_accuracy = accuracy_score(y_true, y_pred)
print("Test accuracy: {}%".format(round(float(test_accuracy), 4) * 100))

# Save ROC results to CSV file.
with open("test_results.csv", "w", newline="") as csvfile:
    result_writger = csv.writer(csvfile)
    result_writger.writerow(["Id", "task_1", "task_2"])
    for test_filepath, test_prediction in zip(test_files, predictions):
        result_writger.writerow([test_filepath, test_prediction[0][0], test_prediction[0][2]])
