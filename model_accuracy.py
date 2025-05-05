from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define validation dataset
val_ds = image_dataset_from_directory(
    "RESISC45/valid",           # path to your validation dataset
    image_size=(224, 224),      # adjust if you used a different size
    batch_size=32,
    shuffle=False               # keep shuffle=False for evaluation/prediction
)

# Load best model from file
best_model = load_model("best_model.h5")

# Evaluate on validation data
val_loss, val_acc = best_model.evaluate(val_ds)
print(f"Best Model - Val Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
