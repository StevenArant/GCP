import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Canvas
import numpy as np
import cv2
import tensorflow as tf  # Correct TensorFlow import
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras import layers, models
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

def create_checkerboard_model(input_shape=(128, 128, 1)):
    model = models.Sequential()

    # Convolutional layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Output layer with two neurons for the center coordinates (x, y)
    model.add(layers.Dense(2, activation='linear'))  # Linear activation for regression

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

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
            img[x_offset:x_offset + checker.shape[0], y_offset:y_offset + checker.shape[1]] = checker
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
    print("Model built successfully.")

def ensure_model_created():
    """Ensures the model is created; builds if it is None."""
    global model
    if model is None:
        print("Model is None, building the model now.")
        model = create_checkerboard_model()  # Call the model creation function
        print("Model built successfully.")

def train_model():
    """Generates data and trains the CNN model."""
    global X_train, y_train, model

    # Ensure the model is created before training
    ensure_model_created()

    # Check if the model is initialized now
    if model is None:
        print("Error: Model could not be built.")
        messagebox.showerror("Error", "Model could not be built. Please check the implementation.")
        return

    # Generate training data
    X_train, y_train, centers = generate_data(1000, 128)
    X_train = X_train[..., np.newaxis] / 255.0  # Normalize and reshape

    # Start training
    try:
        model.fit(X_train, y_train, epochs=5, batch_size=32)
        messagebox.showinfo("Training Complete", "Model training is complete!")
        save_training_data()  # Save the training data after training
    except Exception as e:
        print(f"Error during model training: {e}")
        messagebox.showerror("Training Error", str(e))

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
    canvas_fig = FigureCanvasTkAgg(fig, master=inner_frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack()

    # Configure the scroll region for the inner frame
    inner_frame.update_idletasks()  # Update the inner frame to get its dimensions
    canvas.configure(scrollregion=canvas.bbox("all"))  # Set the scrollable region

def find_checkerboard_center(img):
    """Finds the center of the checkerboard pattern using contour detection."""
    # Convert image to binary (assuming checkerboard is black and white)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours to find the largest one (which should be the checkerboard)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Compute the center of the rectangle (which should be the center of the checkerboard)
        center_x = x + w // 2
        center_y = y + h // 2
        return center_x, center_y

    return None

def upload_and_predict():
    """Uploads an image, predicts the checkerboard center, and displays it."""
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        img_input = img_normalized.reshape((1, 128, 128, 1))  # Reshape for model

        # Make a prediction
        if model is not None:
            prediction = model.predict(img_input)
            center_x, center_y = int(prediction[0][0]), int(prediction[0][1])

            # Draw red 'X' on the predicted center
            img_with_prediction = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(img_with_prediction, (center_x, center_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # Display the predicted image
            plt.imshow(img_with_prediction)
            plt.title(f"Predicted Center: ({center_x}, {center_y})")
            plt.axis('off')
            plt.show()
        else:
            messagebox.showerror("Error", "Model is not trained yet.")

# Load training data and build model on startup
load_training_data()
ensure_model_created()  # Ensure the model is built after loading data

# GUI setup and binding functions to buttons
window = tk.Tk()
window.title("Checkerboard Pattern Detection")

train_button = tk.Button(window, text="Train Model", command=train_model)
train_button.pack()

upload_button = tk.Button(window, text="Upload Image and Predict", command=upload_and_predict)
upload_button.pack()

show_button = tk.Button(window, text="Show Training Images", command=show_training_images)
show_button.pack()

# Start the GUI main loop
window.mainloop()
