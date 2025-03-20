import os
import csv

# Specify the folder path containing the images and depth maps
folder_path = 'data'  # Replace with your folder path
csv_file_path = 'output.csv'  # Specify the output CSV file path

# Get all file names in the folder
file_names = os.listdir(folder_path)

# Filter out all image files (assuming images are in .jpg format)
image_files = [f for f in file_names if f.endswith('.jpg')]

if not image_files:
    print("No image files found in the folder.")
else:
    print(f"Found {len(image_files)} image files.")

# Create and open the CSV file for writing
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Count the number of rows written
    rows_written = 0
    
    # Iterate over all image files and write to CSV
    for image_file in image_files:
        # Replace the image file suffix with the depth map suffix
        depth_file = image_file.replace('.jpg', '_depth.png')
        
        # Check if the depth map exists in the same folder
        if depth_file in file_names:
            # Write to the CSV file with the "sem_images/" prefix
            writer.writerow([f"sem_images/data/{image_file}", f"sem_images/data/{depth_file}"])
            rows_written += 1
        else:
            print(f"Depth file not found for {image_file}")

print(f"CSV file '{csv_file_path}' has been created with {rows_written} rows.")
