# Sample code for predicting and testing multiple sample data
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Set correct image size based on training
IMG_SIZE = (128, 128)

# Load your model
model = load_model("best_model.h5")

# Get class names from folders
DATA_DIR = r"C:\Users\arsul\OneDrive\Desktop\RESISC45\NWPU-RESISC45"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

# ==== 🔁 Provide your 5 specific image paths here ====
img_paths = [
    r"C:\Users\arsul\Desktop\RESISC45\island_700.jpg",
    r"C:\Users\arsul\Desktop\RESISC45\bridge_672.jpg",
    r"C:\Users\arsul\Desktop\RESISC45\dense_residential_680.jpg",
    r"C:\Users\arsul\Desktop\RESISC45\runway_681.jpg",
    r"C:\Users\arsul\Desktop\RESISC45\tennis_court_661.jpg"
]

# ==== 🔍 Predict, display, and save plot ====
plt.figure(figsize=(15, 6))
for i, img_path in enumerate(img_paths):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # No normalization

    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]

    # Plot
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Predicted:\n{predicted_class}", fontsize=10)
    plt.axis("off")

plt.tight_layout()

# Save the figure
output_path = "predictions.png"
plt.savefig(output_path)
plt.show()

print(f"Prediction plot saved as: {output_path}")

