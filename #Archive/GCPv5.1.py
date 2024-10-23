import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, Radiobutton, IntVar, Text, END
from PIL import Image, ImageTk

class ImagePatternMatcher:
    def __init__(self, master):
        self.master = master
        self.master.title("Pattern Matcher")
        self.master.geometry("500x700")

        self.label = Label(master, text="Select an image to search for the pattern.")
        self.label.pack(pady=10)

        self.pattern_preview = Label(master)
        self.pattern_preview.pack(pady=5)

        self.select_pattern_button = Button(master, text="Select Pattern Image", command=self.select_pattern)
        self.select_pattern_button.pack(pady=5)

        self.select_image_button = Button(master, text="Select Target Image", command=self.select_target_image)
        self.select_image_button.pack(pady=5)

        self.threshold_label = Label(master, text="Adjust Certainty Threshold (%):")
        self.threshold_label.pack(pady=5)

        self.threshold_slider = Scale(master, from_=1, to=100, orient=HORIZONTAL)
        self.threshold_slider.set(60)
        self.threshold_slider.pack(pady=5)

        # Radio button selection for different algorithms
        self.algorithm_var = IntVar()
        self.algorithm_var.set(1)  # Default algorithm: ORB

        Label(master, text="Select Matching Algorithm:").pack(pady=5)
        Radiobutton(master, text="ORB", variable=self.algorithm_var, value=1, command=self.update_run_button_state).pack()
        Radiobutton(master, text="AKAZE", variable=self.algorithm_var, value=2, command=self.update_run_button_state).pack()
        Radiobutton(master, text="BRISK", variable=self.algorithm_var, value=3, command=self.update_run_button_state).pack()
        Radiobutton(master, text="Template Matching", variable=self.algorithm_var, value=4, command=self.update_run_button_state).pack()

        self.pattern_image = None
        self.target_image = None
        self.pattern_selected = False
        self.target_selected = False

        # Text field to display match status
        self.match_status = Text(master, height=5, width=50, bg="lightgrey")
        self.match_status.pack(pady=10)

        # Text field to explain algorithms
        self.algorithm_explanation = Text(master, height=10, width=50, bg="lightgrey", wrap='word')
        self.algorithm_explanation.pack(pady=10)

        # Button to run the matching process
        self.run_button = Button(master, text="Run Matching", command=self.process_images, state='disabled')
        self.run_button.pack(pady=10)

        self.display_algorithm_explanations()  # Show algorithm explanations

    def select_pattern(self):
        pattern_path = filedialog.askopenfilename()
        if pattern_path:
            self.pattern_image = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
            self.pattern_selected = True
            self.show_pattern_preview(pattern_path)
            self.select_pattern_button.config(bg='green')  # Set button background to green
            self.update_run_button_state()  # Check if we can enable the run button

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
            self.select_image_button.config(bg='green')  # Set button background to green
            self.update_run_button_state()  # Check if we can enable the run button

    def update_run_button_state(self):
        """Enable or disable the run button based on image loading and selection."""
        if self.pattern_selected and self.target_selected and self.algorithm_var.get() != 0:
            self.run_button.config(state='normal')  # Enable run button
        else:
            self.run_button.config(state='disabled')  # Disable run button

    def process_images(self):
        # Clear old status message
        self.match_status.delete(1.0, END)  # Clear previous status message
        
        if not self.pattern_selected or not self.target_selected:
            return
        
        gray_target_image = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        threshold = self.threshold_slider.get() / 100.0
        
        algorithm = self.algorithm_var.get()
        match_found = False  # Flag to check if any match was found

        if algorithm == 1:
            match_found = self.orb_match(gray_target_image, threshold)
        elif algorithm == 2:
            match_found = self.akaze_match(gray_target_image, threshold)
        elif algorithm == 3:
            match_found = self.brisk_match(gray_target_image, threshold)
        elif algorithm == 4:
            match_found = self.template_matching(gray_target_image, threshold)

        if not match_found:
            self.match_status.insert(END, "No significant match found.")

    # ORB Matching
    def orb_match(self, gray_target_image, threshold):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.pattern_image, None)
        kp2, des2 = orb.detectAndCompute(gray_target_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        return self.display_matches(gray_target_image, kp1, kp2, matches, threshold)

    # AKAZE Matching
    def akaze_match(self, gray_target_image, threshold):
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(self.pattern_image, None)
        kp2, des2 = akaze.detectAndCompute(gray_target_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        return self.display_matches(gray_target_image, kp1, kp2, matches, threshold)

    # BRISK Matching
    def brisk_match(self, gray_target_image, threshold):
        brisk = cv2.BRISK_create()
        kp1, des1 = brisk.detectAndCompute(self.pattern_image, None)
        kp2, des2 = brisk.detectAndCompute(gray_target_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        return self.display_matches(gray_target_image, kp1, kp2, matches, threshold)

    # Template Matching
    def template_matching(self, gray_target_image, threshold):
        best_certainty = 0
        best_location = None
        
        for scale in np.linspace(0.5, 1.5, 10):  # Check multiple scales
            resized_pattern = cv2.resize(self.pattern_image, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(gray_target_image, resized_pattern, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)
            if loc[0].size > 0:
                for pt in zip(*loc[::-1]):
                    best_certainty = result[pt[1], pt[0]]
                    best_location = pt
        
        if best_certainty >= threshold:
            h, w = self.pattern_image.shape[:2]
            cv2.rectangle(gray_target_image, best_location, (best_location[0] + w, best_location[1] + h), (0, 255, 0), 2)
            self.show_image(gray_target_image)
            return True
        
        return False

    def display_matches(self, gray_target_image, kp1, kp2, matches, threshold):
        good_matches = [m for m in matches if m.distance < threshold * 100]  # Using threshold directly
        if len(good_matches) >= threshold * len(matches):
            matched_image = cv2.drawMatches(self.pattern_image, kp1, gray_target_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.show_image(matched_image)
            return True  # Match found
        return False  # No significant match found

    def show_image(self, image):
        cv2.imshow("Matches", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_algorithm_explanations(self):
        explanations = (
            "ORB (Oriented FAST and Rotated BRIEF):\n"
            "A fast and efficient algorithm for detecting and describing keypoints.\n"
            "Invariant to rotation and scale changes, making it suitable for real-time applications.\n\n"
            "AKAZE (Accelerated KAZE):\n"
            "Detects and describes features in non-linear scale spaces.\n"
            "Effective for detecting edges and blobs in images with varied textures.\n\n"
            "BRISK (Binary Robust Invariant Scalable Keypoints):\n"
            "Detects keypoints and descriptors invariant to scale and rotation.\n"
            "Works well in real-time scenarios, ideal for pattern matching.\n\n"
            "Template Matching:\n"
            "A straightforward method that finds a template image in a target image.\n"
            "Effective for detecting exact patterns but less robust to variations."
        )
        self.algorithm_explanation.insert(END, explanations)

# Main GUI loop
if __name__ == "__main__":
    root = Tk()
    matcher = ImagePatternMatcher(root)
    root.mainloop()
