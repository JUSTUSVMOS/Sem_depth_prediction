import os
from PIL import Image
import numpy as np

def histogram_correction(image):
    # Convert the image to grayscale
    gray_image = image.convert("L")
    
    # Convert the image to a numpy array
    image_array = np.array(gray_image)
    
    # Compute the thresholds for the lowest 0.2% and highest 0.2%
    low_percentile = np.percentile(image_array, 0.2)
    high_percentile = np.percentile(image_array, 99.8)
    
    # Set pixels below the low threshold to the minimum value and above the high threshold to the maximum value
    image_array = np.clip(image_array, low_percentile, high_percentile)
    
    # Scale the pixel values to the range 0-255
    image_array = ((image_array - low_percentile) / (high_percentile - low_percentile) * 255).astype(np.uint8)
    
    # Convert the processed array back to a PIL image
    corrected_image = Image.fromarray(image_array)
    
    return corrected_image

def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    for filename in os.listdir(input_folder):
        # Process only .jpg and .jpeg files
        if filename.lower().endswith(('.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Open the image and perform histogram correction
            with Image.open(input_path) as img:
                corrected_img = histogram_correction(img)
                # Save the processed image
                corrected_img.save(output_path)
                print(f"Processed and saved: {output_path}")

# Set the input and output folder paths
input_folder = './original_data'  # Replace with your input folder path
output_folder = './original_data'  # Replace with your output folder path

# Process all .jpg and .jpeg images in the folder
process_folder(input_folder, output_folder)
