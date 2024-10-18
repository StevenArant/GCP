import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, messagebox
from PIL import Image, ImageTk

class ImagePatternMatcher:
    def __init__(self, master):
        self.master = master
        self.master.title("Pattern Matcher")
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
        self.threshold_label = Label(master, text="Adjust Certainty Threshold (%):")
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
        
        # Get the threshold value from the slider
        threshold = self.threshold_slider.get() / 100.0  # Convert slider value to a decimal
        
        # Multi-scale template matching to account for different sizes
        best_match = None
        best_certainty = 0
        best_location = None
        best_scale = 1.0
        
        # Try matching with different scales of the pattern image
        for scale in np.linspace(0.5, 2.0, 20):  # Test different scales from 50% to 200%
            scaled_pattern = self.resize_image(self.pattern_image, scale)
            if gray_target_image.shape[0] < scaled_pattern.shape[0] or gray_target_image.shape[1] < scaled_pattern.shape[1]:
                continue  # Skip if the scaled pattern is larger than the target image
            
            res = cv2.matchTemplate(gray_target_image, scaled_pattern, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_certainty:  # Track the best match found
                best_certainty = max_val
                best_match = scaled_pattern
                best_location = max_loc
                best_scale = scale
        
        # Proceed only if best_certainty exceeds the threshold
        if best_certainty >= threshold:
            # Draw the smaller red "X" at the best match point
            x1, y1 = best_location
            x2, y2 = x1 + best_match.shape[1], y1 + best_match.shape[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw a smaller red "X" at the best match point (half the size of the original)
            cv2.line(self.target_image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Line 1 (smaller)
            cv2.line(self.target_image, (x1, y2), (x2, y1), (0, 0, 255), 1)  # Line 2 (smaller)

            # Certainty percentage calculation
            certainty = best_certainty * 100

            # Calculate the size difference
            original_size = self.pattern_image.shape
            matched_size = best_match.shape
            size_difference = (matched_size[1] / original_size[1]) * 100  # Width comparison
            text = f'{certainty:.2f}% certainty | Size: {size_difference:.2f}%'

            # Determine the size of the text for placing the black background
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw a black rectangle as a background for the text
            top_left = (best_location[0], best_location[1] - text_height - 10)
            bottom_right = (best_location[0] + text_width, best_location[1])
            cv2.rectangle(self.target_image, top_left, bottom_right, (0, 0, 0), -1)  # Filled black rectangle

            # Now draw the text on top of the black rectangle
            cv2.putText(self.target_image, text, (best_location[0], best_location[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show the result in a window
        self.show_image(self.target_image)
    
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
