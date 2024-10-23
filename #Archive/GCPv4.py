import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, messagebox
from PIL import Image, ImageTk

class ImagePatternMatcher:
    def __init__(self, master):
        self.master = master
        self.master.title("Pattern Matcher with ORB")
        self.master.geometry("400x500")
        
        self.label = Label(master, text="Select an image to search for the pattern.")
        self.label.pack(pady=10)
        
        self.pattern_preview = Label(master)
        self.pattern_preview.pack(pady=5)
        
        self.select_pattern_button = Button(master, text="Select Pattern Image", command=self.select_pattern)
        self.select_pattern_button.pack(pady=5)

        self.select_image_button = Button(master, text="Select Target Image", command=self.select_target_image)
        self.select_image_button.pack(pady=5)
        
        # Slider for adjusting the certainty threshold
        self.threshold_label = Label(master, text="Adjust Match Confidence Threshold:")
        self.threshold_label.pack(pady=5)

        self.threshold_slider = Scale(master, from_=1, to=100, orient=HORIZONTAL)
        self.threshold_slider.set(60)  # Default threshold set to 60%
        self.threshold_slider.pack(pady=5)

        self.pattern_image = None
        self.target_image = None
        self.pattern_selected = False
        self.target_selected = False

    def select_pattern(self):
        pattern_path = filedialog.askopenfilename()
        if pattern_path:
            self.pattern_image = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
            self.pattern_selected = True
            # Show a preview of the pattern image in the GUI
            self.show_pattern_preview(pattern_path)
            messagebox.showinfo("Success", "Pattern image selected!")
    
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

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and descriptors in both pattern and target images
        keypoints_pattern, descriptors_pattern = orb.detectAndCompute(self.pattern_image, None)
        keypoints_target, descriptors_target = orb.detectAndCompute(gray_target_image, None)

        # Initialize BFMatcher with KNN
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descriptors_pattern, descriptors_target, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)

        # Get the threshold value from the slider
        threshold = self.threshold_slider.get() / 100.0  # Convert slider value to a decimal
        num_good_matches = len(good_matches)  # Number of good matches found

        if num_good_matches > 10:  # Ensure a minimum of 10 good matches
            # Draw matches
            match_result = cv2.drawMatches(self.pattern_image, keypoints_pattern, gray_target_image, keypoints_target, good_matches, None, flags=2)

            # Compute Homography if enough good matches are found
            src_pts = np.float32([keypoints_pattern[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = self.pattern_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw lines around the detected pattern in the target image
            gray_target_image = cv2.polylines(gray_target_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Draw the best match percentage on the image
            best_match_distance = good_matches[0].distance  # Distance of the closest match
            max_possible_distance = 256  # Max distance for ORB
            match_percentage = (1 - (best_match_distance / max_possible_distance)) * 100

            # Position the text near the first match point
            x, y = int(dst[0][0][0]), int(dst[0][0][1])  # Coordinates of the first point in the detected pattern
            text = f"Match: {match_percentage:.2f}%"

            # Draw a black rectangle as background for the text
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(gray_target_image, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)

            # Put the text on the image
            cv2.putText(gray_target_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show the result in a window
            self.show_image(gray_target_image)
        else:
            messagebox.showinfo("No Match", "No significant match found. Try adjusting the threshold or using a different pattern.")

    def show_image(self, img):
        cv2.imshow("Matched Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main GUI loop
if __name__ == "__main__":
    root = Tk()
    matcher = ImagePatternMatcher(root)
    root.mainloop()
