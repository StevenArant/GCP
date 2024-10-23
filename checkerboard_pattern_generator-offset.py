import numpy as np
from PIL import ImageFilter, Image

def generate_shifted_checkerboard(size, max_blur):
  """
  Generates a fuzzy 2x2 checkerboard image with a randomly shifted center,
  ensuring it stays within the specified coordinates without scaling.

  Args:
    size: The size of the image in pixels.
    max_blur: The maximum blur radius.

  Returns:
    A PIL Image object.
  """

  # Create a base checkerboard pattern
  pattern = np.array([[0, 255], [255, 0]])
  pattern_img = Image.fromarray(pattern)

  # Calculate the maximum shift to keep the pattern center within (0, 0) and (size, size)
  max_shift = size // 2 - 1

  # Randomly choose a starting position within the bounds
  start_x = np.random.randint(max_shift, size - max_shift)
  start_y = np.random.randint(max_shift, size - max_shift)

  # Create a blank image
  img = Image.new("L", (size, size), 255)

  # Create a grayscale mask with full opacity (all white)
  mask = pattern_img.convert("L")
  mask = mask.point(lambda p: 255)

  # Paste the pattern at the chosen position without scaling
  img.paste(pattern_img, (start_x, start_y), mask=mask)

  # Apply random Gaussian blur
  blur_radius = np.random.randint(0, max_blur + 1)
  img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

  return img

# Generate a collection of shifted fuzzy images
for i in range(20):
  img = generate_shifted_checkerboard(128, 60)
  img.save(f"shifted_checkerboard_{i}.png")