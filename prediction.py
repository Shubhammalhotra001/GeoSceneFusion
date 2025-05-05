#Code for checking 1 sample image 
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Set correct image size based on training
IMG_SIZE = (128, 128)

# Load your model
model = load_model("best_model.h5")

# Get class names from folders
DATA_DIR = r"C:\Users\arsul\Desktop\RESISC45\NWPU-RESISC45"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

# Load and preprocess the image
img_path = r"C:\Users\arsul\Desktop\RESISC45\ball-courts.jpg"
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
preds = model.predict(img_array)
predicted_class = CLASS_NAMES[np.argmax(preds)]
print(f"Predicted class: {predicted_class}")
