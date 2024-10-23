import cv2
import numpy as np
import os
import pandas as pd
from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, Text, END, Toplevel, messagebox, Scrollbar, Frame
from PIL import Image, ImageTk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ImagePatternMatcher:
    def __init__(self, master):
        self.master = master
        self.master.title("Template Pattern Matcher")
        self.master.geometry("600x700")

        self.label = Label(master, text="Select an image to search for the pattern.")
        self.label.pack(pady=10)

        self.pattern_preview = Label(master)
        self.pattern_preview.pack(pady=5)

        self.select_pattern_button = Button(master, text="Select Pattern Image", command=self.select_pattern)
        self.select_pattern_button.pack(pady=5)

        self.select_image_button = Button(master, text="Select Target Image", command=self.select_target_image)
        self.select_image_button.pack(pady=5)

        self.run_button = Button(master, text="Run Matching", command=self.process_images, state="disabled")
        self.run_button.pack(pady=5)

        # Slider for adjusting the certainty threshold
        self.threshold_label = Label(master, text="Adjust Certainty Threshold (%):")
        self.threshold_label.pack(pady=5)

        self.threshold_slider = Scale(master, from_=1, to=100, orient=HORIZONTAL)
        self.threshold_slider.set(60)  # Default threshold set to 60%
        self.threshold_slider.pack(pady=5)

        self.info_box = Text(master, height=10, width=68)
        self.info_box.pack(pady=5)

        self.pattern_image = None
        self.target_image = None
        self.pattern_selected = False
        self.target_selected = False

        self.match_candidates = []  # To store candidates for user selection
        self.feedback_data_file = "feedback_data.csv"
        self.explanation_text = (
            "Template Matching:\n"
            "This method compares a pattern image against a target image to find the best match\n"
            "by computing similarity using normalized cross-correlation. It can handle\n"
            "rotations, scaling, and translations to improve accuracy."
        )

        # Load feedback data
        self.feedback_data = self.load_feedback_data()
        self.model = self.train_model()

        self.display_explanation()  # Display explanation in the info box

    def select_pattern(self):
        pattern_path = filedialog.askopenfilename()
        if pattern_path:
            self.pattern_image = cv2.imread(pattern_path, cv2.IMREAD_UNCHANGED)
            self.pattern_selected = True
            self.show_pattern_preview(pattern_path)
            self.update_buttons()

    def show_pattern_preview(self, pattern_path):
        pattern_image = Image.open(pattern_path)
        pattern_image.thumbnail((150, 150))  # Resize for preview
        pattern_img_tk = ImageTk.PhotoImage(pattern_image)
        self.pattern_preview.config(image=pattern_img_tk)
        self.pattern_preview.image = pattern_img_tk

    def select_target_image(self):
        target_path = filedialog.askopenfilename()
        if target_path:
            self.target_image = cv2.imread(target_path)
            self.target_selected = True
            self.update_buttons()

    def update_buttons(self):
        if self.pattern_selected and self.target_selected:
            self.run_button.config(state="normal")
            self.select_pattern_button.config(bg="green")
            self.select_image_button.config(bg="green")
        else:
            self.run_button.config(state="disabled")

    def process_images(self):
        # Clear previous match candidates
        self.match_candidates.clear()

        if not self.pattern_selected or not self.target_selected:
            messagebox.showerror("Error", "Please select both pattern and target images!")
            return

        # Clear previous results in the text box
        self.info_box.delete('1.0', END)

        # Get the threshold value from the slider
        threshold = self.dynamic_threshold_adjustment()

        # Pre-process target image: enhance contrast
        enhanced_target = self.enhance_contrast(self.target_image)

        # Perform multi-scale template matching using Gaussian pyramid
        pattern_pyramid = self.create_gaussian_pyramid(self.pattern_image)

        for scale_pattern in pattern_pyramid:
            res = cv2.matchTemplate(enhanced_target, scale_pattern, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:  # Check against threshold
                self.match_candidates.append((max_val, max_loc, scale_pattern))

        # Analyze feedback and refine matches if applicable
        self.analyze_feedback()

        # Display matches for user selection
        self.display_matches_grid()

    def dynamic_threshold_adjustment(self):
        """Adjust threshold dynamically based on user feedback."""
        if not os.path.exists(self.feedback_data_file):
            return self.threshold_slider.get() / 100.0  # Default threshold if no feedback

        feedback_df = pd.read_csv(self.feedback_data_file, header=None, names=["location", "score"])
        if feedback_df.empty:
            return self.threshold_slider.get() / 100.0  # Default threshold if no feedback

        # Calculate average score
        avg_score = feedback_df["score"].mean()
        adjusted_threshold = max(0.01, avg_score)  # Ensure threshold is not below a certain level
        return adjusted_threshold

    def analyze_feedback(self):
        """Analyze feedback data to refine matching strategy."""
        if not os.path.exists(self.feedback_data_file):
            return

        feedback_df = pd.read_csv(self.feedback_data_file, header=None, names=["location", "score"])

        if feedback_df.empty:
            return

        # Train the machine learning model based on feedback
        self.model = self.train_model(feedback_df)

    def train_model(self, feedback_df=None):
        """Train a simple logistic regression model using feedback data."""
        if feedback_df is None:
            feedback_df = pd.read_csv(self.feedback_data_file, header=None, names=["location", "score"])

        if feedback_df.empty:
            return None

        # Map feedback to binary classes
        feedback_df['label'] = feedback_df['score'].apply(lambda x: 1 if x > 0.6 else 0)

        X = feedback_df[['score']]  # Feature: score
        y = feedback_df['label']  # Label: whether the score indicates success

        # Check if we have at least two classes in the dataset
        if len(y.unique()) < 2:
            print("Not enough classes to train the model. Using default behavior.")
            return None  # Return None to indicate model cannot be trained

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Print accuracy of the model
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")

        return model

    def display_matches_grid(self):
        if not self.match_candidates:
            self.info_box.insert(END, "No matches found.")
            return

        # Create a new window for the grid of matches
        grid_window = Toplevel(self.master)
        grid_window.title("Select Best Match")

        # Create a frame for scrollable area
        frame = Frame(grid_window)
        frame.pack(fill="both", expand=True)

        # Create a canvas to hold the scrollable content
        canvas = canvas(frame)
        scroll_y = Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_y.pack(side="right", fill="y")

        # Create a window in the canvas to hold the images
        self.match_frame = Frame(canvas)
        canvas.create_window((0, 0), window=self.match_frame, anchor="nw")

        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scroll_y.set)

        # Add the canvas to the window
        canvas.pack(side="left", fill="both", expand=True)

        # Create a label for each match in the grid
        for idx, (score, loc, scale_pattern) in enumerate(self.match_candidates[:6]):  # Show top 6 matches
            certainty = score * 100
            match_image = self.draw_rectangle(self.target_image.copy(), loc, scale_pattern.shape[1::-1])
            match_image_pil = Image.fromarray(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
            match_image_tk = ImageTk.PhotoImage(match_image_pil)

            match_label = Label(self.match_frame, image=match_image_tk, text=f"{certainty:.2f}%", compound='top')
            match_label.image = match_image_tk  # Keep a reference to avoid garbage collection
            match_label.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)

            # Bind click event to the label
            match_label.bind("<Button-1>", lambda event, idx=idx: self.store_feedback(self.match_candidates[idx], match_label))

        # Update scroll region
        self.match_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        no_best_match_button = Button(grid_window, text="No Suitable Match", command=lambda: self.store_feedback(None, None))
        no_best_match_button.pack(pady=10)

    def draw_rectangle(self, image, top_left, dimensions):
        """Draw a rectangle on the image based on match location."""
        bottom_right = (top_left[0] + dimensions[0], top_left[1] + dimensions[1])
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle
        return image

    def store_feedback(self, match, label):
        """Store feedback for matches."""
        if match:
            max_val, loc, scale_pattern = match
            feedback_value = messagebox.askyesno("Feedback", "Was this match suitable?")
            with open(self.feedback_data_file, "a") as file:
                file.write(f"{loc}, {int(feedback_value)}\n")  # Store location and feedback
            if feedback_value:
                self.info_box.insert(END, f"Stored feedback for match at {loc}. \n")
            else:
                self.info_box.insert(END, "Match marked as unsuitable.\n")
        else:
            self.info_box.insert(END, "No suitable match selected.\n")

    def enhance_contrast(self, image):
        """Enhance contrast of the input image using histogram equalization."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.equalizeHist(gray_image)
        return cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    def create_gaussian_pyramid(self, image, levels=3):
        """Create a Gaussian pyramid of the image."""
        pyramid = [image]
        for _ in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def load_feedback_data(self):
        """Load feedback data from a CSV file."""
        if os.path.exists(self.feedback_data_file):
            return pd.read_csv(self.feedback_data_file, header=None, names=["location", "score"])
        return pd.DataFrame(columns=["location", "score"])

    def display_explanation(self):
        """Display explanation of the matching method."""
        self.info_box.insert(END, self.explanation_text)

if __name__ == "__main__":
    root = Tk()
    app = ImagePatternMatcher(root)
    root.mainloop()
