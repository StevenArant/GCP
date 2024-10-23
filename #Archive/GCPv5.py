import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, Radiobutton, IntVar, messagebox
from PIL import Image, ImageTk

class ImagePatternMatcher:
    def __init__(self, master):
        self.master = master
        self.master.title("Pattern Matcher")
        self.master.geometry("400x600")
        
        self.label = Label(master, text="Select an image to search for the pattern.")
        self.label.pack(pady=10)
        
        self.pattern_preview = Label(master)
        self.pattern_preview.pack(pady=5)
        
        self.select_pattern_button = Button(master, text="Select Pattern Image", command=self.select_pattern)
        self.select_pattern_button.pack(pady=5)

        self.select_image_button = Button(master, text="Select Target Image", command=self.select_target_image)
        self.select_image_button.pack(pady=5)

        # Slider for adjusting the certainty threshold
        self.threshold_label = Label(master, text="Adjust Certainty Threshold (%):")
        self.threshold_label.pack(pady=5)

        self.threshold_slider = Scale(master, from_=1, to=100, orient=HORIZONTAL)
        self.threshold_slider.set(60)  # Default threshold set to 60%
        self.threshold_slider.pack(pady=5)

        self.pattern_image = None
        self.target_image = None
        self.pattern_selected = False
        self.target_selected = False

        # Radio buttons for algorithm selection
        self.algorithms_label = Label(master, text="Select Pattern Recognition Algorithm:")
        self.algorithms_label.pack(pady=5)

        self.algorithm_var = IntVar()
        self.algorithm_var.set(1)  # Default to ORB

        self.orb_button = Radiobutton(master, text="ORB (Oriented FAST and Rotated BRIEF)", variable=self.algorithm_var, value=1)
        self.orb_button.pack(anchor="w")

        self.sift_button = Radiobutton(master, text="SIFT (Scale-Invariant Feature Transform)", variable=self.algorithm_var, value=2)
        self.sift_button.pack(anchor="w")

        self.template_button = Radiobutton(master, text="Template Matching", variable=self.algorithm_var, value=3)
        self.template_button.pack(anchor="w")
    
    def select_pattern(self):
        pattern_path = filedialog.askopenfilename()
        if pattern_path:
            self.pattern_image = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
            self.pattern_selected = True
            self.show_pattern_preview(pattern_path)
    
    def show_pattern_preview(self, pattern_path):
        # Load the image using PIL to display in the Tkinter GUI
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
            
            if self.pattern_selected and self.target_selected:
                self.process_images()
            else:
                messagebox.showerror("Error", "Please select both pattern and target images!")

    def process_images(self):
        if not self.pattern_selected or not self.target_selected:
            messagebox.showerror("Error", "Please select both pattern and target images!")
            return
        
        gray_target_image = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

        # Get the selected algorithm
        selected_algorithm = self.algorithm_var.get()

        # Call the corresponding method based on the selected algorithm
        if selected_algorithm == 1:
            self.orb_match(gray_target_image)
        elif selected_algorithm == 2:
            self.sift_match(gray_target_image)
        elif selected_algorithm == 3:
            self.template_match(gray_target_image)
    
    def orb_match(self, gray_target_image):
        # ORB-based pattern matching
        orb = cv2.ORB_create()
        keypoints_pattern, descriptors_pattern = orb.detectAndCompute(self.pattern_image, None)
        keypoints_target, descriptors_target = orb.detectAndCompute(gray_target_image, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_pattern, descriptors_target)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get threshold from slider
        threshold = self.threshold_slider.get() / 100.0
        num_good_matches = int(len(matches) * threshold)

        if len(matches) > 10:
            # Draw matches and compute homography
            match_result = cv2.drawMatches(self.pattern_image, keypoints_pattern, gray_target_image, keypoints_target, matches[:num_good_matches], None, flags=2)
            self.show_image(match_result)
        else:
            messagebox.showinfo("No Match", "No significant match found with ORB.")

    def sift_match(self, gray_target_image):
        # SIFT-based pattern matching
        sift = cv2.SIFT_create()
        keypoints_pattern, descriptors_pattern = sift.detectAndCompute(self.pattern_image, None)
        keypoints_target, descriptors_target = sift.detectAndCompute(gray_target_image, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors_pattern, descriptors_target)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get threshold from slider
        threshold = self.threshold_slider.get() / 100.0
        num_good_matches = int(len(matches) * threshold)

        if len(matches) > 10:
            # Draw matches and show result
            match_result = cv2.drawMatches(self.pattern_image, keypoints_pattern, gray_target_image, keypoints_target, matches[:num_good_matches], None, flags=2)
            self.show_image(match_result)
        else:
            # Using messagebox for proper error handling
            messagebox.showinfo("No Match", "No significant match found with SIFT.")

    def template_match(self, gray_target_image):
        # Template Matching (already implemented)
        threshold = self.threshold_slider.get() / 100.0
        best_match = None
        best_certainty = 0
        best_location = None

        for scale in np.linspace(0.4, 2.0, 20):
            scaled_pattern = self.resize_image(self.pattern_image, scale)
            if gray_target_image.shape[0] < scaled_pattern.shape[0] or gray_target_image.shape[1] < scaled_pattern.shape[1]:
                continue
            
            res = cv2.matchTemplate(gray_target_image, scaled_pattern, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_certainty:
                best_certainty = max_val
                best_match = scaled_pattern
                best_location = max_loc

        if best_certainty >= threshold:
            # Draw result and certainty percentage
            x1, y1 = best_location
            x2, y2 = x1 + best_match.shape[1], y1 + best_match.shape[0]
            cv2.rectangle(self.target_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"{best_certainty * 100:.2f}%"
            cv2.putText(self.target_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.show_image(self.target_image)
        else:
            messagebox.showinfo("No Match", "No significant match found with Template Matching.")

    def resize_image(self, image, scale):
        """Resize an image by a given scale factor."""
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        return cv2.resize(image, (width, height))

    def show_image(self, img):
        cv2.imshow("Matched Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main GUI loop
if __name__ == "__main__":
    root = Tk()
    matcher = ImagePatternMatcher(root)
    root.mainloop()
