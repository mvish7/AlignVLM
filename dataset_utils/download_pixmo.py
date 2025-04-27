import os
import requests
from datasets import load_dataset, Dataset, Features, Value # No longer need HFImage here directly for saving
from PIL import UnidentifiedImageError # Still useful for potential checks, but not the primary goal
from PIL import Image
import io
import time
import uuid # For generating unique filenames
import pathlib # For easier path manipulation
import mimetypes # For guessing file extensions from MIME types

# --- Configuration ---
DATASET_NAME = "allenai/pixmo-cap"
SPLIT = "train"
SIZE = 0.35  # i.e. 10%, use number/100.
IMAGE_SAVE_DIR = "/media/vishal/datasets/pixmo/images/" # Directory to save image files
OUTPUT_DIR = "/media/vishal/datasets/pixmo/metadata/" # Directory to save the processed dataset (with paths)
NUM_PROC = os.cpu_count()
REQUEST_TIMEOUT = 10
RETRY_ATTEMPTS = 2
RETRY_DELAY = 2

# --- Ensure Image Save Directory Exists ---
IMAGE_SAVE_PATH = pathlib.Path(IMAGE_SAVE_DIR)
IMAGE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
print(f"Images will be saved to: {IMAGE_SAVE_PATH.resolve()}")

# --- 1. Load the Initial Dataset ---
print(f"\nLoading initial dataset: {DATASET_NAME} (split: {SPLIT})")
try:
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    ds = ds.shuffle().select(range(int(SIZE*len(ds))))
    if 'image_url' not in ds.features:
        raise ValueError(f"'image_url' column not found in dataset {DATASET_NAME}. Available columns: {list(ds.features.keys())}")
except Exception as e:
    print(f"Error loading initial dataset: {e}")
    exit()

print(f"Initial dataset loaded with {len(ds)} samples.")
if len(ds) == 0:
    print("Warning: Loaded dataset is empty.")
    exit()


# --- 2. Define the Processing Function (Downloads to Disk) ---
def download_and_save_image(example):
    """
    Downloads image from URL, saves it to disk, and returns the file path.
    Returns None for 'image_path' if download or saving fails.
    """
    image_url = example.get('image_url')
    if not image_url:
        return {**example, 'image_path': None, 'download_status': 'missing_url'}

    file_path = None # Initialize file_path to None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(image_url, timeout=REQUEST_TIMEOUT, stream=True) # Use stream=True
            response.raise_for_status() # Check for HTTP errors

            # --- Generate Filename and Path ---
            # Get content type to guess extension
            content_type = response.headers.get('content-type')
            extension = mimetypes.guess_extension(content_type) if content_type else '.jpg' # Default to .jpg if unknown
            # if not extension or extension == '.jpe': # Handle common variations or lack of guess
            #     extension = '.jpg'

            # Create unique filename using UUID
            filename = f"{uuid.uuid4()}{extension}"
            file_path = IMAGE_SAVE_PATH / filename # Use pathlib for joining path

            while file_path.exists(): # Ensure unique filename
                filename = f"{uuid.uuid4()}{extension}"
                file_path = IMAGE_SAVE_PATH / filename

            # --- Stream download to file ---
            try:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): # Download in chunks
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                # If download and save successful
                return {**example, 'image_path': str(file_path.resolve()), 'download_status': 'success'}
            except IOError as e_io:
                # print(f"Error saving file {file_path} for URL {image_url}: {e_io}")
                # Clean up potentially partially written file
                if file_path.exists():
                    try:
                       file_path.unlink()
                    except OSError:
                        pass # Ignore cleanup error if file couldn't be deleted
                return {**example, 'image_path': None, 'download_status': f'file_io_error: {e_io}'}

        except requests.exceptions.Timeout:
            # print(f"Warning: Timeout downloading {image_url} (Attempt {attempt + 1}/{RETRY_ATTEMPTS})")
            status = 'timeout'
            if attempt + 1 < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
                continue # Go to next retry attempt
            else:
                 return {**example, 'image_path': None, 'download_status': status}
        except requests.exceptions.RequestException as e:
            # print(f"Warning: Failed to download {image_url}: {e} (Attempt {attempt + 1}/{RETRY_ATTEMPTS})")
            status = f'request_error: {e}'
            if attempt + 1 < RETRY_ATTEMPTS:
                 time.sleep(RETRY_DELAY)
                 continue # Go to next retry attempt
            else:
                 return {**example, 'image_path': None, 'download_status': status}
        except Exception as e_global: # Catch any other unexpected errors
             # print(f"Warning: Unexpected error for {image_url}: {e_global}")
             return {**example, 'image_path': None, 'download_status': f'unexpected_error: {e_global}'}

    # Fallback if loop finishes without success (shouldn't happen with current logic)
    return {**example, 'image_path': None, 'download_status': 'failed_after_retries'}

# --- 4. Apply the Function using map ---
print(f"\nProcessing dataset using {NUM_PROC} cores (saving images to disk)...")

# Define the features of the new dataset. 'image_path' will store the string path.
new_features = ds.features.copy()
new_features['image_path'] = Value('string') # Store the path as a string
new_features['download_status'] = Value('string')

processed_ds = ds.map(
    download_and_save_image,
    num_proc=NUM_PROC,
    features=new_features,
    batched=False,
    # remove_columns=['image_url'] # Optional
)

print("Processing complete.")

# --- 5. Handle Errors / Filter Failed Downloads ---
print("\nFiltering out failed downloads...")
original_count = len(processed_ds)
# Keep only examples where image_path is not None (i.e., download and save succeeded)
processed_ds = processed_ds.filter(lambda example: example['image_path'] is not None and example['download_status'] == 'success')

print("\nFiltering out corrupted images")
corrupted_id = []
small_res_id = []
high_res_ids = []
high_res_limit = 2048 * 2048

for sample_id, sample in enumerate(processed_ds):
    try:
        curr_img = Image.open(sample["image_path"])
        w, h = curr_img.size
        mode = curr_img.mode
        if w < 50 or h < 50 or mode not in ['RGB', 'RGBA', 'YCbCr', "P", "L", "CMYK", "LA", "1"]:
            print(f"dropped image due to {w}x{h} size or {mode} mode")
            small_res_id.append(sample_id)
        # elif w * h >  high_res_limit:
        #     print(f"dropped image due to {w}x{h} size")
        #     high_res_ids.append(sample_id)

    except:  # if image is corrupted or cannot be opened
        print(f"Corrupted image: {sample['image_path']}")
        corrupted_id.append(sample_id)

all_ids_to_remove = set(corrupted_id + small_res_id)
# Remove images from the dataset
final_ds = processed_ds.select(
    i for i in range(len(processed_ds)) if i not in all_ids_to_remove
)

print(f"Removed {len(corrupted_id)} corrupted images.")
print(f"Removed {len(small_res_id)} small resolution images.")
print(f"Removed {len(high_res_ids)} high resolution images.")

failed_count = original_count - len(final_ds)
print(f"Removed total of {failed_count} examples.")
print(f"Final dataset size: {len(final_ds)} samples.")

# 5.1 removing super high resolution images, which cause out of memory issues
# Remove images with extremely high resolution (e.g., > 10,000,000 pixels)
# high_res_ids = []
# for sample_id, sample in enumerate(final_ds):
#     try:
#         curr_img = Image.open(sample["image_path"])
#         w, h = curr_img.size
#         if w * h > 10000000:  # Example threshold
#             print(f"dropped image due to {w}x{h} size")
#             high_res_ids.append(sample_id)
#     except UnidentifiedImageError:
#         print(f"UnidentifiedImageError for image {sample['image_path']}")
# Remove high resolution images from the dataset
# final_ds = final_ds.select(

# --- 6. Save the Processed Dataset (with Paths) ---
print(f"\nSaving processed dataset metadata to disk: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Remove the status column before saving
final_ds = final_ds.remove_columns(['download_status'])
print("Final dataset features before saving:", final_ds.features)

final_ds.save_to_disk(OUTPUT_DIR)
print("Dataset metadata saved successfully.")
print(f"Image files are stored separately in: {IMAGE_SAVE_PATH.resolve()}")
