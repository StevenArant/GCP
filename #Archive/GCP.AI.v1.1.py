import cv2
import numpy as np
import os
import pandas as pd  # For handling feedback data
from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, Text, END, Toplevel, messagebox
from PIL import Image, ImageTk
from collections import defaultdict

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
        self.display_explanation()

        # Load feedback data
        self.feedback_data = self.load_feedback_data()

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
        threshold = self.threshold_slider.get() / 100.0  # Convert slider value to a decimal

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

    def analyze_feedback(self):
        """Analyze feedback data to refine matching strategy."""
        if not os.path.exists(self.feedback_data_file):
            return

        feedback_df = pd.read_csv(self.feedback_data_file, header=None, names=["location", "score"])
        score_counts = defaultdict(int)

        for index, row in feedback_df.iterrows():
            location = row['location']
            score_counts[location] += 1

        # Adjust matching strategy based on feedback trends
        # For simplicity, here we will print trends. In a real implementation, we might adjust parameters.
        for location, count in score_counts.items():
            print(f"Location {location} selected {count} times.")

    def display_matches_grid(self):
        if not self.match_candidates:
            self.info_box.insert(END, "No matches found.")
            return

        # Create a new window for the grid of matches
        grid_window = Toplevel(self.master)
        grid_window.title("Select Best Match")

        # Create a label for each match in the grid
        for idx, (score, loc, scale_pattern) in enumerate(self.match_candidates[:6]):  # Show top 6 matches
            certainty = score * 100
            match_image = self.draw_rectangle(self.target_image.copy(), loc, scale_pattern.shape[1::-1])
            match_image_pil = Image.fromarray(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
            match_image_tk = ImageTk.PhotoImage(match_image_pil)

            match_label = Label(grid_window, image=match_image_tk, text=f"{certainty:.2f}%", compound='top')
            match_label.image = match_image_tk  # Keep a reference to avoid garbage collection
            match_label.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)

            # Bind click event to the label
            match_label.bind("<Button-1>", lambda event, idx=idx: self.store_feedback(self.match_candidates[idx], match_label))
        
        no_best_match_button = Button(grid_window, text="No Best Match", command=grid_window.destroy)
        no_best_match_button.grid(row=2, column=0, columnspan=3, pady=10)

        grid_window.mainloop()

    def store_feedback(self, selected_match, match_label):
        """Store user feedback for learning."""
        score, loc, scale_pattern = selected_match
        feedback_data = f"{loc},{score}\n"

        # Write feedback to the CSV file
        with open(self.feedback_data_file, "a") as f:
            f.write(feedback_data)

        self.flash_selection(match_label)
        self.info_box.insert(END, "Feedback recorded!\n")

        # Optionally, analyze feedback again to see impact on next matches
        self.analyze_feedback()

    def flash_selection(self, match_label):
        """Flash the selected match label to indicate selection."""
        original_bg = match_label.cget("bg")
        match_label.config(bg="yellow")  # Change to yellow
        match_label.after(500, lambda: match_label.config(bg=original_bg))  # Revert back after 500ms

    def enhance_contrast(self, image):
        """Enhance contrast of the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_color = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        return enhanced_color

    def create_gaussian_pyramid(self, image, levels=5):
        """Create a Gaussian pyramid for the given image."""
        pyramid = [image]
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def draw_rectangle(self, img, top_left, dimensions):
        """Draw a rectangle on the image."""
        cv2.rectangle(img, top_left, (top_left[0] + dimensions[0], top_left[1] + dimensions[1]), (0, 255, 0), 2)
        return img

    def show_image(self, img):
        cv2.imshow("Matched Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_feedback_data(self):
        """Load feedback data from CSV file."""
        if os.path.exists(self.feedback_data_file):
            return pd.read_csv(self.feedback_data_file, header=None)
        return pd.DataFrame(columns=["location", "score"])

    def display_explanation(self):
        self.info_box.insert(END, self.explanation_text)

if __name__ == "__main__":
    root = Tk()
    app = ImagePatternMatcher(root)
    root.mainloop()
