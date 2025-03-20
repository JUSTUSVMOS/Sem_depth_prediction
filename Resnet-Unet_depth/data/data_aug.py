import os
import glob
import numpy as np
from PIL import Image

def random_translate_scale(img, depth_img, scale_min=1.0, scale_max=1.2):
    """
    Apply random scaling and translation to both the RGB and Depth images (applied once),
    returning the new (padded_img, padded_depth).
    - The scale factor is chosen from the range [scale_min, scale_max]
    - Translation range: ±(original width/height // 4)
    """
    original_width, original_height = img.size

    # Random scaling factor
    scale_factor = np.random.uniform(scale_min, scale_max)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Scale the RGB image using LANCZOS and the Depth image using NEAREST
    scaled_img = img.resize((new_width, new_height), Image.LANCZOS)
    depth_img = depth_img.convert("L")  # Ensure depth is in grayscale mode
    scaled_depth = depth_img.resize((new_width, new_height), Image.NEAREST)

    # Create a blank canvas with the same size as the original image
    padded_img = Image.new('RGB', (original_width, original_height), (0, 0, 0))
    # For depth maps, use a single channel with a gray background (89) to represent invalid values
    padded_depth = Image.new('L', (original_width, original_height), 89)

    # Random translation
    x_shift = np.random.randint(-original_width // 4, original_width // 4)
    y_shift = np.random.randint(-original_height // 4, original_height // 4)

    x_start = max(0, min((original_width - new_width) // 2 + x_shift, original_width - new_width))
    y_start = max(0, min((original_height - new_height) // 2 + y_shift, original_height - new_height))

    # Paste the scaled images onto the canvas
    padded_img.paste(scaled_img, (x_start, y_start))
    padded_depth.paste(scaled_depth, (x_start, y_start))

    return padded_img, padded_depth

def random_resize_crop(img, depth_img, crop_ratio):
    """
    Randomly crop both images by the specified crop_ratio, then resize back to the original size.
    The crop is applied once, ensuring that the RGB and Depth images remain synchronized.
    """
    img_width, img_height = img.size
    crop_width = int(img_width * crop_ratio)
    crop_height = int(img_height * crop_ratio)

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("The crop ratio is too small, leading to an invalid crop size.")

    # Randomly determine the top-left corner of the crop
    x_start = np.random.randint(0, img_width - crop_width + 1)
    y_start = np.random.randint(0, img_height - crop_height + 1)

    # Crop both the RGB and Depth images
    cropped_img = img.crop((x_start, y_start, x_start + crop_width, y_start + crop_height))
    cropped_depth = depth_img.crop((x_start, y_start, x_start + crop_width, y_start + crop_height))

    # Resize the cropped images back to the original size
    resized_cropped_img = cropped_img.resize((img_width, img_height), Image.LANCZOS)
    resized_cropped_depth = cropped_depth.resize((img_width, img_height), Image.NEAREST)

    return resized_cropped_img, resized_cropped_depth

def single_augment_and_save(img, depth_img, output_dir, base_name, index,
                            scale_min=1.0, scale_max=1.2, crop_ratio=0.08):
    """
    For a given pair of (RGB, Depth) images, perform one round of augmentation
    (random scaling/translation + one crop) and output one augmented pair.
    """
    # Step 1: Random scaling and translation
    translated_img, translated_depth = random_translate_scale(
        img, depth_img, scale_min=scale_min, scale_max=scale_max
    )

    # Step 2: Random crop (applied once)
    final_img, final_depth = random_resize_crop(translated_img, translated_depth, crop_ratio)

    # Create the output directory if it doesn't exist and save the files
    os.makedirs(output_dir, exist_ok=True)

    output_img_path = os.path.join(output_dir, f"{base_name}_{index}.jpg")
    output_depth_path = os.path.join(output_dir, f"{base_name}_{index}_depth.png")

    final_img.save(output_img_path)
    final_depth.save(output_depth_path)

    print(f"Saved final augmented pair:\n  {output_img_path}\n  {output_depth_path}")

def process_folder(input_dir, output_dir, num_images=1, scale_min=1.0, scale_max=1.2,
                   crop_ratio=0.08):
    """
    Traverse all .jpg files in the folder, and find the corresponding *_depth.png files.
    For each (RGB, Depth) pair, generate num_images augmented pairs, each outputting one pair.
    """
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        base_name = os.path.basename(img_path).replace('.jpg', '')
        depth_img_path = os.path.join(input_dir, f'{base_name}_depth.png')

        if not os.path.exists(depth_img_path):
            print(f"[Warning] Corresponding depth map not found: {depth_img_path}, skipping this image.")
            continue

        # Read the original image and depth map
        img = Image.open(img_path).convert('RGB')    # Ensure RGB format
        depth_img = Image.open(depth_img_path).convert('L')  # Maintain single channel

        # For each image, perform augmentation num_images times (each generating one pair)
        for i in range(num_images):
            single_augment_and_save(
                img, depth_img,
                output_dir=output_dir,
                base_name=base_name,
                index=f"{i+1}",
                scale_min=scale_min,
                scale_max=scale_max,
                crop_ratio=crop_ratio
            )

# --------------------------- Main Program -------------------------------------
if __name__ == "__main__":
    # Parameter settings
    input_dir = "./original_data"       # Original RGB and Depth images in the same folder
    output_dir = "./data"  # Output directory for augmented results
    num_images = 5             # Number of augmented versions to generate per image
    scale_min = 1.0
    scale_max = 1.2
    crop_ratio = 0.8          # Random crop ratio

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        num_images=num_images,
        scale_min=scale_min,
        scale_max=scale_max,
        crop_ratio=crop_ratio
    )
