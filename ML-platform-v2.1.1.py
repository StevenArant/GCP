import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog, Canvas
import numpy as np
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#from tensorflow import keras
import tensorflow as tf; tf.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random


# Paths for saving the training data
X_train_path = "X_train.npy"
y_train_path = "y_train.npy"

# Global variables for training data and model
model = None
X_train = np.array([])  # Training images
y_train = np.array([])  # Training labels

def create_checkerboard_model(input_shape=(128, 128, 1)):
    """Creates a CNN model for checkerboard detection."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Output layer for (x, y)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def load_training_data():
    """Loads training data from disk if available."""
    global X_train, y_train
    if os.path.exists(X_train_path) and os.path.exists(y_train_path):
        X_train = np.load(X_train_path, allow_pickle=True)
        y_train = np.load(y_train_path, allow_pickle=True)
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
            
            # Append the center (x, y) as a 2D label [x, y]
            label = [x_offset + checker.shape[0] // 2, y_offset + checker.shape[1] // 2]
            assert len(label) == 2, f"Label must have two values, but got {label}"  # Check if label has 2 values
            labels.append(label)
            centers.append((x_offset + checker.shape[0] // 2, y_offset + checker.shape[1] // 2))
        else:
            # Non-checkerboard case, append a placeholder label like [0, 0]
            label = [0, 0]
            assert len(label) == 2, f"Non-checkerboard label must have two values, but got {label}"  # Check
            labels.append(label)
            centers.append(None)
        data.append(img)
    
    # Convert to NumPy arrays
    data_array = np.array(data)
    labels_array = np.array(labels)

    # Check final label shapes
    assert labels_array.shape[1] == 2, f"Each label must have two values, but found shape {labels_array.shape}"
    
    return data_array, labels_array, centers



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

    # Generate training data
    X_train, y_train, centers = generate_data(1000, 128)
    X_train = X_train[..., np.newaxis] / 255.0  # Normalize and reshape

    # Ensure y_train contains an even number of elements
    if len(y_train) % 2 != 0:
        raise ValueError(f"y_train has an incorrect number of elements: {len(y_train)}")

    # Reshape y_train to ensure it's 2D with shape (n_samples, 2)
    y_train = np.array(y_train).reshape(-1, 2)

    # Check shapes
    print("X_train shape:", X_train.shape)  # Should be (n_samples, img_size, img_size, 1)
    print("y_train shape:", y_train.shape)  # Should be (n_samples, 2)

    # Compile the model with the correct loss function
    model.compile(optimizer='adam', loss='mean_squared_error')

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
            cv2.drawMarker(img_with_prediction, (center_x, center_y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=2)

            # Show image with prediction
            plt.imshow(img_with_prediction, cmap='gray')
            plt.title(f"Predicted center: ({center_x}, {center_y})")
            plt.show()

            # Ask user for feedback
            feedback = messagebox.askyesno("Prediction", f"Is the predicted center correct? ({center_x}, {center_y})")
            if not feedback:
                # Ask user for the correct center
                correct_x = simpledialog.askinteger("Correction", "Enter the correct X coordinate:", minvalue=0, maxvalue=127)
                correct_y = simpledialog.askinteger("Correction", "Enter the correct Y coordinate:", minvalue=0, maxvalue=127)

                # Update the training data with the corrected label
                global X_train, y_train
                if X_train.size == 0:
                    X_train = img_input
                    y_train = np.array([[correct_x, correct_y]])
                else:
                    X_train = np.vstack([X_train, img_input])
                    y_train = np.vstack([y_train, [correct_x, correct_y]])

                # Save the updated training data
                save_training_data()
                messagebox.showinfo("Update", "Training data updated with the correct center.")

def main_window():
    """Creates the main window with buttons and options."""
    global window
    window = tk.Tk()
    window.title("Checkerboard Detector")

    # Create buttons for various functions
    btn_train = tk.Button(window, text="Train Model", command=train_model)
    btn_train.pack(pady=10)

    btn_show_images = tk.Button(window, text="Show Training Images", command=show_training_images)
    btn_show_images.pack(pady=10)

    btn_upload_predict = tk.Button(window, text="Upload and Predict", command=upload_and_predict)
    btn_upload_predict.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    load_training_data()
    main_window()
