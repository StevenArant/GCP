import numpy as np
from PIL import ImageFilter, Image

def generate_fuzzy_checkerboard(size, max_blur):
  """
  Generates a fuzzy 2x2 checkerboard image with varying blur levels.

  Args:
    size: The size of the image in pixels.
    max_blur: The maximum blur radius.

  Returns:
    A PIL Image object.
  """

  # Create a base checkerboard pattern
  pattern = np.array([[0, 255], [255, 0]])
  pattern = np.repeat(pattern, size // 2, axis=0)
  pattern = np.repeat(pattern, size // 2, axis=1)

  # Normalize contrast to prevent clipping
  pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255

  # Convert to PIL Image
  img = Image.fromarray(pattern.astype(np.uint8))

  # Apply random Gaussian blur
  blur_radius = np.random.randint(0, max_blur + 1)
  img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

  return img

# Generate a collection of fuzzy images
for i in range(20):
  img = generate_fuzzy_checkerboard(128, 60)
  img.save(f"fuzzy_checkerboard_{i}.png")