import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Global model variable
model = None

def create_checkerboard(size):
    """Generates a 2x2 checkerboard pattern"""
    pattern = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    return cv2.resize(pattern, (size, size))

def generate_data(n_samples, img_size):
    data, labels = [], []
    for _ in range(n_samples):
        img = np.random.rand(img_size, img_size) * 255
        if np.random.random() > 0.5:
            checker = create_checkerboard(np.random.randint(10, 50))
            x_offset = np.random.randint(0, img_size - checker.shape[0])
            y_offset = np.random.randint(0, img_size - checker.shape[1])
            img[x_offset:x_offset+checker.shape[0], y_offset:y_offset+checker.shape[1]] = checker
            labels.append(1)
        else:
            labels.append(0)
        data.append(img)
    return np.array(data), np.array(labels)

def build_model():
    """Builds and returns a simple CNN model."""
    global model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def train_model():
    """Generates data and trains the CNN model"""
    X_train, y_train = generate_data(1000, 64)
    X_train = X_train[..., np.newaxis] / 255.0  # Normalize and reshape
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    messagebox.showinfo("Training Complete", "Model training is complete!")

def show_training_images():
    """Displays a few images used for training."""
    X_train, y_train = generate_data(9, 64)
    X_train = X_train[..., np.newaxis] / 255.0  # Normalize and reshape

    fig, axs = plt.subplots(3, 3, figsize=(5, 5))
    for i, ax in enumerate(axs.flat):
        ax.imshow(X_train[i].reshape(64, 64), cmap='gray')
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis('off')

    # Create the figure on the Tkinter GUI
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def upload_and_predict():
    """Opens file dialog to allow users to upload an image for prediction."""
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (64, 64))
        img_resized = img_resized[np.newaxis, ..., np.newaxis] / 255.0
        prediction = model.predict(img_resized)
        label = "Checkerboard Detected" if prediction > 0.5 else "No Checkerboard Detected"
        messagebox.showinfo("Prediction Result", label)

# Initialize the GUI window
window = tk.Tk()
window.title("Checkerboard Pattern Recognition")

# Add a few buttons to control the process
btn_train = tk.Button(window, text="Train Model", command=train_model)
btn_train.pack(pady=10)

btn_show_images = tk.Button(window, text="Show Training Images", command=show_training_images)
btn_show_images.pack(pady=10)

btn_upload = tk.Button(window, text="Upload Image for Prediction", command=upload_and_predict)
btn_upload.pack(pady=10)

# Build the CNN model at start
build_model()

# Start the GUI loop
window.mainloop()
