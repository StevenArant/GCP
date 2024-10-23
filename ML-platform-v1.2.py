import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Canvas
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random
import os

# Paths for saving the training data
X_train_path = "X_train.npy"
y_train_path = "y_train.npy"

# Global variables for training data and model
model = None
X_train = np.array([])  # Training images
y_train = np.array([])  # Training labels

def load_training_data():
    """Loads training data from disk if available."""
    global X_train, y_train
    if os.path.exists(X_train_path) and os.path.exists(y_train_path):
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        print("Training data loaded.")
    else:
        X_train, y_train = np.array([]), np.array([])  # Initialize empty if not found
        print("No existing training data found, starting fresh.")

def save_training_data():
    """Saves training data to disk."""
    np.save(X_train_path, X_train)
    np.save(y_train_path, y_train)
    print("Training data saved.")

def create_checkerboard(size, angle=0):
    """Generates a 2x2 checkerboard pattern with random rotation."""
    pattern = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    pattern_resized = cv2.resize(pattern, (size, size))

    # Randomly rotate the pattern
    if angle != 0:
        center = (pattern_resized.shape[1] // 2, pattern_resized.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        pattern_resized = cv2.warpAffine(pattern_resized, matrix, (size, size))

    return pattern_resized

def generate_data(n_samples, img_size):
    """Generates randomized checkerboard and non-checkerboard images."""
    data, labels, centers = [], [], []
    for _ in range(n_samples):
        img = np.random.rand(img_size, img_size) * 255
        if np.random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            checker = create_checkerboard(np.random.randint(20, 60), angle)
            x_offset = np.random.randint(0, img_size - checker.shape[0])
            y_offset = np.random.randint(0, img_size - checker.shape[1])
            img[x_offset:x_offset+checker.shape[0], y_offset:y_offset+checker.shape[1]] = checker
            labels.append(1)
            centers.append((x_offset + checker.shape[0] // 2, y_offset + checker.shape[1] // 2))
        else:
            labels.append(0)
            centers.append(None)
        data.append(img)
    return np.array(data), np.array(labels), centers

def build_model():
    """Builds and returns a CNN model with higher resolution filters."""
    global model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def train_model():
    """Generates data and trains the CNN model."""
    global X_train, y_train
    X_train, y_train, centers = generate_data(1000, 128)
    X_train = X_train[..., np.newaxis] / 255.0  # Normalize and reshape
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    messagebox.showinfo("Training Complete", "Model training is complete!")
    save_training_data()  # Save the training data after training

def show_training_images():
    """Displays a few images used for training with a red 'X' at checkerboard center."""
    X_sample, y_sample, centers = generate_data(9, 128)
    X_sample = X_sample[..., np.newaxis] / 255.0  # Normalize and reshape

    # Create a scrollable frame for the images
    scrollable_frame = tk.Frame(window)
    scrollable_frame.pack(fill=tk.BOTH, expand=True)

    canvas = Canvas(scrollable_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(scrollable_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Frame for placing the images inside the canvas
    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    # Ensure that window is resizable
    window.geometry("800x600")
    window.minsize(400, 300)

    fig, axs = plt.subplots(3, 3, figsize=(5, 5))
    for i, ax in enumerate(axs.flat):
        ax.imshow(X_sample[i].reshape(128, 128), cmap='gray')
        if centers[i] is not None:
            ax.plot(centers[i][1], centers[i][0], 'rx', markersize=12, mew=2)  # Red 'X'
        ax.set_title(f"Label: {y_sample[i]}")
        ax.axis('off')

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=inner_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def upload_and_predict():
    """Opens file dialog to allow users to upload an image for prediction."""
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (128, 128))
        img_resized = img_resized[np.newaxis, ..., np.newaxis] / 255.0
        prediction = model.predict(img_resized)
        label = "Checkerboard Detected" if prediction > 0.5 else "No Checkerboard Detected"
        
        # Simulate finding the center (for demo purposes, use the middle of the image)
        center_x, center_y = 64, 64  # Assuming the center is the middle of the resized image
        
        # Display the image with a red 'X' marking the checkerboard center
        fig, ax = plt.subplots()
        ax.imshow(img_resized[0].reshape(128, 128), cmap='gray')
        if prediction > 0.5:
            ax.plot(center_x, center_y, 'rx', markersize=12, mew=2)
        ax.set_title(f"Prediction: {label}")
        ax.axis('off')
        plt.show()

        # Show prediction and ask for confirmation
        result = messagebox.askyesnocancel("Prediction Result", f"{label}. Is the location correct?")
        
        if result is None:
            # If the user cancels, do nothing
            return
        elif result:
            # If the user confirms both pattern presence and location
            add_to_training(img_resized, 1)
        else:
            # Ask if the pattern was present but the location was incorrect
            incorrect_location = messagebox.askyesno("Correction", "Is the checkerboard present but location incorrect?")
            if incorrect_location:
                # Retrain with corrected location information
                add_to_training(img_resized, 1)
            else:
                add_to_training(img_resized, 0)

def add_to_training(image, label):
    """Adds the confirmed/rejected image to the training set and retrains the model."""
    global X_train, y_train
    if X_train.size == 0:  # If no training data exists yet
        X_train = image
        y_train = np.array([label])
    else:
        X_train = np.vstack([X_train, image])
        y_train = np.hstack([y_train, label])

    # Retrain the model with new data
    model.fit(X_train, y_train, epochs=1, batch_size=32)
    messagebox.showinfo("Update Complete", "Model retrained with new data!")

    # Save the updated training data to disk
    save_training_data()

# Initialize the GUI window
window = tk.Tk()
window.title("Checkerboard Pattern Recognition")

# Add buttons for control
btn_train = tk.Button(window, text="Train Model", command=train_model)
btn_train.pack(pady=10)

btn_show_images = tk.Button(window, text="Show Training Images", command=show_training_images)
btn_show_images.pack(pady=10)

btn_upload = tk.Button(window, text="Upload Image for Prediction", command=upload_and_predict)
btn_upload.pack(pady=10)

# Build the CNN model at start
build_model()

# Load the training data at the start of the program
load_training_data()

# Start the GUI loop
window.mainloop()
