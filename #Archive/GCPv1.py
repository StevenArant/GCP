import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button
from tkinter import messagebox
import os

# Setting default thickness and font scale via environment variables for flexibility
TEXT_THICKNESS = int(os.getenv("TEXT_THICKNESS", 2))
TEXT_SCALE = float(os.getenv("TEXT_SCALE", 0.5))
CIRCLE_THICKNESS = int(os.getenv("CIRCLE_THICKNESS", 2))
LINE_THICKNESS = int(os.getenv("LINE_THICKNESS", 2))


class ImagePatternMatcher:
    def __init__(self, master):
        self.master = master
        self.master.title("Pattern Matcher")
        self.master.geometry("400x200")
        
        self.label = Label(master, text="Select an image to search for the pattern.")
        self.label.pack(pady=10)
        
        self.select_pattern_button = Button(master, text="Select Pattern Image", command=self.select_pattern)
        self.select_pattern_button.pack(pady=5)
        
        self.select_image_button = Button(master, text="Select Target Image", command=self.select_target_image)
        self.select_image_button.pack(pady=5)
        
        self.pattern_image = None
        self.target_image = None
    
    def select_pattern(self):
        pattern_path = filedialog.askopenfilename()
        if pattern_path:
            self.pattern_image = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
            messagebox.showinfo("Success", "Pattern image selected!")
    
    def select_target_image(self):
        target_path = filedialog.askopenfilename()
        if target_path:
            self.target_image = cv2.imread(target_path)
            self.process_images()
    
    def process_images(self):
        if self.pattern_image is None or self.target_image is None:
            messagebox.showerror("Error", "Please select both pattern and target images!")
            return
        
        gray_target_image = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        res = cv2.matchTemplate(gray_target_image, self.pattern_image, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6  # Set to match patterns with a similarity of 60% or higher
        loc = np.where(res >= threshold)

        # Draw red circle and 'X' on matching points and display certainty percentage
        for pt in zip(*loc[::-1]):
            # Draw a red "X"
            x1, y1 = pt[0], pt[1]
            x2, y2 = pt[0] + self.pattern_image.shape[1], pt[1] + self.pattern_image.shape[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            cv2.line(self.target_image, (x1, y1), (x2, y2), (0, 0, 255), LINE_THICKNESS)
            cv2.line(self.target_image, (x1, y2), (x2, y1), (0, 0, 255), LINE_THICKNESS)

            # Draw a red circle at the center of the found pattern
            radius = int(min(self.pattern_image.shape[:2]) // 2)
            cv2.circle(self.target_image, (center_x, center_y), radius, (0, 0, 255), CIRCLE_THICKNESS)

            # Certainty percentage calculation
            certainty = res[pt[1], pt[0]] * 100
            text = f'{certainty:.2f}%'

            # Determine the size of the text for placing the black background
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)

            # Draw a black rectangle as a background for the text
            top_left = (pt[0], pt[1] - text_height - 10)
            bottom_right = (pt[0] + text_width, pt[1])
            cv2.rectangle(self.target_image, top_left, bottom_right, (0, 0, 0), -1)  # Filled black rectangle

            # Now draw the text on top of the black rectangle
            cv2.putText(self.target_image, text, (pt[0], pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 255), TEXT_THICKNESS)

        # Show the result in a window
        self.show_image(self.target_image)
    
    def show_image(self, img):
        cv2.imshow("Matched Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main GUI loop
if __name__ == "__main__":
    root = Tk()
    matcher = ImagePatternMatcher(root)
    root.mainloop()
