from keras.applications.inception_v3 import preprocess_input
from keras.applications import InceptionV3
from keras.models import model_from_json
from glob import glob
from utils import *
import argparse
import cv2

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
    help="Path to the input image")
args = vars(ap.parse_args())

# Input image.
input_image = args['image']

""" Transfer learning using Inception V3 """
# Load the Inception V3 model as well as the network weights from disk.
print("[INFO] loading {}...".format("CNN Model"))
transfer_model = InceptionV3(include_top=False, weights="imagenet")

""" Retrieve the saved CNN model """
# Load json and create model.
json_file = open('models/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model.
loaded_model.load_weights("weights/CNN_model.h5")
CNN_model = loaded_model
print("[INFO] Loaded model from the disk.")

# Prediction.
tensor = paths_to_tensor([input_image])
preprocessed_image = preprocess_input(tensor)
feature_extracted_image = transfer_model.predict(preprocessed_image)
prediction = CNN_model.predict(feature_extracted_image)

# Load lables.
label_name = [item[11:-1] for item in sorted(glob("data/train/*/"))]
# print("[INFO] Label names are: {}".format(label_name))
print("[INFO] Analyzing the skin lesion.")
print("[INFO] Please Wait...")

# Show output.
cv2.namedWindow('Classification', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Classification', 1920, 1080)
orig = cv2.imread(args["image"])
label_index = np.argmax(prediction)
label = label_name[label_index]
prob = prediction[0][label_index]
print("[INFO] Analysis Completed!")
print("[INFO] {} detected in the image.".format(label))
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(50, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)