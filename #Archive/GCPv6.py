import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, Text, END, messagebox
from PIL import Image, ImageTk

class ImagePatternMatcher:
    def __init__(self, master):
        self.master = master
        self.master.title("Template Pattern Matcher")
        self.master.geometry("400x600")
        
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

        self.info_box = Text(master, height=10, width=48)
        self.info_box.pack(pady=5)

        self.pattern_image = None
        self.target_image = None
        self.pattern_selected = False
        self.target_selected = False

        self.explanation_text = (
            "Template Matching:\n"
            "This method compares a pattern image against a target image to find the best match\n"
            "by computing similarity using normalized cross-correlation. It can handle\n"
            "rotations, scaling, and translations to improve accuracy."
        )
        self.display_explanation()

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
        if not self.pattern_selected or not self.target_selected:
            messagebox.showerror("Error", "Please select both pattern and target images!")
            return
        
        # Clear previous results in the text box
        self.info_box.delete('1.0', END)

        # Get the threshold value from the slider
        threshold = self.threshold_slider.get() / 100.0  # Convert slider value to a decimal

        # Multi-scale template matching to account for different sizes
        best_match = None
        best_certainty = 0
        best_location = None
        
        # Try matching with different scales of the pattern image
        for scale in np.linspace(0.5, 1.5, 20):  # Test different scales
            scaled_pattern = self.resize_image(self.pattern_image, scale)
            if self.target_image.shape[0] < scaled_pattern.shape[0] or self.target_image.shape[1] < scaled_pattern.shape[1]:
                continue  # Skip if the scaled pattern is larger than the target image
            
            # Use cv2.matchTemplate for template matching
            res = cv2.matchTemplate(self.target_image, scaled_pattern, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_certainty:  # Track the best match found
                best_certainty = max_val
                best_match = scaled_pattern
                best_location = max_loc

        # Proceed only if best_certainty exceeds the threshold
        if best_certainty >= threshold:
            # Draw the red "X" at the best match point
            x1, y1 = best_location
            x2, y2 = x1 + best_match.shape[1], y1 + best_match.shape[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw a red "X" at the best match point 
            cv2.line(self.target_image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Line 1 
            cv2.line(self.target_image, (x1, y2), (x2, y1), (0, 0, 255), 1)  # Line 2 

            # Certainty percentage calculation
            certainty = best_certainty * 100
            text = f'{certainty:.2f}% certainty'

            # Draw text on the target image
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(self.target_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)  # Background rectangle
            cv2.putText(self.target_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.show_image(self.target_image)
        else:
            self.info_box.insert(END, "No significant match found.")

    def resize_image(self, image, scale):
        """Resize an image by a given scale factor."""
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        return cv2.resize(image, (width, height))

    def show_image(self, img):
        cv2.imshow("Matched Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_explanation(self):
        self.info_box.insert(END, self.explanation_text)

# Main GUI loop
if __name__ == "__main__":
    root = Tk()
    matcher = ImagePatternMatcher(root)
    root.mainloop()
