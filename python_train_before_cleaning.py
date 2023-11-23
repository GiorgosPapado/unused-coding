# %%
import os
import datasets
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor,AutoTokenizer
os.environ["WANDB_DISABLED"] = "true"


# %%
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# %%
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

image_encoder_model = "Centaur31/vit-base"
text_decode_model = "gpt2"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    image_encoder_model, text_decode_model)

# %%
# image feature extractor
feature_extractor = AutoImageProcessor.from_pretrained(image_encoder_model)
# text tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_decode_model)

# %%
# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# %%
model.encoder.embeddings.patch_embeddings.projection

# %%
output_dir = "vit-gpt-model"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
import os

def list_items_in_folder(folder_path):
    items_list = []

    # Iterate over each item (files or directories) in the folder
    for item in os.listdir(folder_path):
        # Get the full path of the item
        item_path = os.path.join(folder_path, item)
        # Append the item's path to the list
        items_list.append(item_path)

    return items_list

# Replace 'folder_path' with the path to the folder you want to read
folder_path = '/home/vcl3d/coco_dataset_VOX_5000/train2014'
items_list = list_items_in_folder(folder_path)
items_list = sorted(items_list)
# Print the list of items in the folder
#print(items_list)

# %%
#EVALUATION 
import os

def list_items_in_folder(folder_path):
    items_list = []

    # Iterate over each item (files or directories) in the folder
    for item in os.listdir(folder_path):
        # Get the full path of the item
        item_path = os.path.join(folder_path, item)
        # Append the item's path to the list
        items_list.append(item_path)

    return items_list

# Replace 'folder_path' with the path to the folder you want to read
val_folder_path = '/home/vcl3d/coco_dataset_VOX_5000/val2014'
val_items_list = list_items_in_folder(val_folder_path)
val_items_list=sorted(val_items_list)

# Print the list of items in the folder
print(val_items_list)

# %%
# #DEPTH
# depth_path = '/home/vcl3d/coco_dataset_VOX_mini/train2014_d'
# depth_list = list_items_in_folder(depth_path)
# depth_list = sorted(depth_list)
# print(depth_list)

# %%
# #VAL_DEPTH
# val_depth_path = '/home/vcl3d/coco_dataset_VOX_mini/val2014_d'
# val_depth_list = list_items_in_folder(val_depth_path)
# val_depth_list = sorted(val_depth_list)
# print(val_depth_list)

# %%
from datasets import load_dataset, Image, Dataset
data = {"image": items_list}
# Step 3: Convert the list to a Dataset object
dataset = Dataset.from_dict(data)

# Step 4: Cast the "image" column to the Image() type
dataset = dataset.cast_column("image", Image())

# %%
# #DEPTH
# depth_data = {"image": depth_list}
# # Step 3: Convert the list to a Dataset object
# depth_dataset = Dataset.from_dict(depth_data)

# # Step 4: Cast the "image" column to the Image() type
# depth_dataset = depth_dataset.cast_column("image", Image())


# %%
# #VAL_DEPTH
# val_depth_data = {"image": val_depth_list}
# # Step 3: Convert the list to a Dataset object
# val_depth_dataset = Dataset.from_dict(val_depth_data)

# # Step 4: Cast the "image" column to the Image() type
# val_depth_dataset = val_depth_dataset.cast_column("image", Image())

# %%
#EVALUATION
from datasets import load_dataset, Image, Dataset
val_data = {"image": val_items_list}
# Step 3: Convert the list to a Dataset object
val_dataset = Dataset.from_dict(val_data)

# Step 4: Cast the "image" column to the Image() type
val_dataset = val_dataset.cast_column("image", Image())

# %%
heights_list = []
widths_list = []
image_id_counter = 0
text_id_counter = 0
image_id_list = []
text_id_list = []
# Loop through the 'image' column of the dataset
for image in dataset['image']:
    # Get the height and width of the current image
    height, width = image.size

    # Append the height and width to their respective lists
    heights_list.append(height)
    widths_list.append(width)
    image_id = image_id_counter
    image_id_list.append(image_id)
    image_id_counter += 1
# Print the lists of heights and widths
#print("List of Heights:", heights_list)
#print("List of Widths:", widths_list)

# %%
#EVALUATION
val_heights_list = []
val_widths_list = []
val_image_id_counter = 0
val_text_id_counter = 0
val_image_id_list = []
val_text_id_list = []
# Loop through the 'image' column of the dataset
for image in val_dataset['image']:
    # Get the height and width of the current image
    val_height, val_width = image.size

    # Append the height and width to their respective lists
    val_heights_list.append(val_height)
    val_widths_list.append(val_width)
    val_image_id = val_image_id_counter
    val_image_id_list.append(val_image_id)
    val_image_id_counter += 1
# Print the lists of heights and widths
#print("List of Heights:", val_heights_list)
#print("List of Widths:", val_widths_list)

# %%
# #DEPTH
# depth_heights_list = []
# depth_widths_list = []
# depth_image_id_counter = 0
# depth_text_id_counter = 0
# depth_image_id_list = []
# depth_text_id_list = []
# # Loop through the 'image' column of the dataset
# for image in depth_dataset['image']:
#     # Get the height and width of the current image
#     depth_height, depth_width = image.size

#     # Append the height and width to their respective lists
#     depth_heights_list.append(depth_height)
#     depth_widths_list.append(depth_width)
#     depth_image_id = depth_image_id_counter
#     depth_image_id_list.append(depth_image_id)
#     depth_image_id_counter += 1
# # Print the lists of heights and widths
# print("List of Heights:", depth_heights_list)
# print("List of Widths:", depth_widths_list)

# %%
# #VAL_DEPTH
# val_depth_heights_list = []
# val_depth_widths_list = []
# val_depth_image_id_counter = 0
# val_depth_text_id_counter = 0
# val_depth_image_id_list = []
# val_depth_text_id_list = []
# # Loop through the 'image' column of the dataset
# for image in val_depth_dataset['image']:
#     # Get the height and width of the current image
#     val_depth_height, val_depth_width = image.size

#     # Append the height and width to their respective lists
#     val_depth_heights_list.append(val_depth_height)
#     val_depth_widths_list.append(val_depth_width)
#     val_depth_image_id = val_depth_image_id_counter
#     val_depth_image_id_list.append(val_depth_image_id)
#     val_depth_image_id_counter += 1
# # Print the lists of heights and widths
# print("List of Heights:", val_depth_heights_list)
# print("List of Widths:", val_depth_widths_list)

# %%
import re
def remove_numbers(text_descriptions):
    clean_text_descriptions = []
    for line in text_descriptions:
        clean_text_descriptions.append((re.sub(r'\d+', '', line)).strip())  # Remove digits and leading/trailing spaces
    combined_text = '. '.join(clean_text_descriptions) + '.'  # Add a period at the end
    return combined_text



# %%
def limit_words(text, word_limit=200):
    words = text.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + '...'  # Add ellipsis if text is truncated
    return text

# %%
# #DEPTH
# text_files_path = '/home/vcl3d/coco_dataset_VOX_mini/train2014_desc'
# text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
# text_files = sorted(text_files)
# #text_files = sorted(text_files[:100])
# # Create a dictionary to store text descriptions with image filenames (without extension) as keys
# text_dict = {}

# # Load text descriptions from each text file and match them with the images
# for text_file in text_files:
#     image_name = os.path.splitext(text_file)[0]
#     text_file_path = os.path.join(text_files_path, text_file)

#     with open(text_file_path, 'r') as file:
#         text_descriptions = file.read().splitlines()

#     text_dict[image_name] = text_descriptions

#     text_descriptions = remove_numbers(text_descriptions)
#     text_dict[image_name] = text_descriptions
# # Convert image paths to strings by extracting the file name from the full path
# image_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] for image_path in depth_dataset["image"]]
# text_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] + '_desc'  for image_path in dataset["image"]]
# # Add the "text" column to the dimg dataset
# depth_dataset = depth_dataset.add_column("text", [text_dict[filename] for filename in text_filenames])
# depth_dataset = depth_dataset.add_column("image_path", depth_list)
# depth_dataset = depth_dataset.add_column("height", depth_heights_list)
# depth_dataset = depth_dataset.add_column("width", depth_widths_list)
# depth_dataset = depth_dataset.add_column("image_id", depth_image_id_list)
# #dataset = dataset.add_column("caption_id", text_id_list)
# # Print the resulting dataset with image paths and text descriptions
# #text_dict['COCO_train2014_000000000049_desc']
# depth_dataset = depth_dataset.add_column("file_name", image_filenames)

# %%
# #VAL_DEPTH
# text_files_path = '/home/vcl3d/coco_dataset_VOX_mini/val2014_desc'
# text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
# text_files = sorted(text_files)
# #text_files = sorted(text_files[:100])
# # Create a dictionary to store text descriptions with image filenames (without extension) as keys
# text_dict = {}

# # Load text descriptions from each text file and match them with the images
# for text_file in text_files:
#     image_name = os.path.splitext(text_file)[0]
#     text_file_path = os.path.join(text_files_path, text_file)

#     with open(text_file_path, 'r') as file:
#         text_descriptions = file.read().splitlines()

#     text_dict[image_name] = text_descriptions

#     text_descriptions = remove_numbers(text_descriptions)
#     text_dict[image_name] = text_descriptions
# # Convert image paths to strings by extracting the file name from the full path
# image_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] for image_path in val_depth_dataset["image"]]
# text_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] + '_desc'  for image_path in val_dataset["image"]]
# # Add the "text" column to the dimg dataset
# val_depth_dataset = val_depth_dataset.add_column("text", [text_dict[filename] for filename in text_filenames])
# val_depth_dataset = val_depth_dataset.add_column("image_path", val_depth_list)
# val_depth_dataset = val_depth_dataset.add_column("height", val_depth_heights_list)
# val_depth_dataset = val_depth_dataset.add_column("width", val_depth_widths_list)
# val_depth_dataset = val_depth_dataset.add_column("image_id", val_depth_image_id_list)
# #dataset = dataset.add_column("caption_id", text_id_list)
# # Print the resulting dataset with image paths and text descriptions
# #text_dict['COCO_train2014_000000000049_desc']
# val_depth_dataset = val_depth_dataset.add_column("file_name", image_filenames)

# %%
text_files_path = '/home/vcl3d/coco_dataset_VOX_5000/train2014_desc'
text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
text_files = sorted(text_files)
#text_files = sorted(text_files[:100])
# Create a dictionary to store text descriptions with image filenames (without extension) as keys
text_dict = {}

# Load text descriptions from each text file and match them with the images
for text_file in text_files:
    image_name = os.path.splitext(text_file)[0]
    text_file_path = os.path.join(text_files_path, text_file)

    with open(text_file_path, 'r') as file:
        text_descriptions = file.read().splitlines()

    text_dict[image_name] = text_descriptions

    text_descriptions = remove_numbers(text_descriptions)
    text_descriptions = limit_words(text_descriptions , word_limit=50)
    text_dict[image_name] = text_descriptions
# Convert image paths to strings by extracting the file name from the full path
image_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] for image_path in dataset["image"]]
text_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] + '_desc'  for image_path in dataset["image"]]
# Add the "text" column to the dimg dataset
dataset = dataset.add_column("caption", [text_dict[filename] for filename in text_filenames])
dataset = dataset.add_column("image_path", items_list)
dataset = dataset.add_column("height", heights_list)
dataset = dataset.add_column("width", widths_list)
dataset = dataset.add_column("image_id", image_id_list)
#dataset = dataset.add_column("caption_id", text_id_list)
# Print the resulting dataset with image paths and text descriptions
#text_dict['COCO_train2014_000000000049_desc']
dataset = dataset.add_column("file_name", image_filenames)

# %%
#EVALUATION
val_text_files_path = '/home/vcl3d/coco_dataset_VOX_5000/val2014_desc'
val_text_files = [file for file in os.listdir(val_text_files_path) if file.endswith('_desc.txt')]
val_text_files = sorted(val_text_files)
#val_text_files = sorted(val_text_files[:100])

# Create a dictionary to store text descriptions with image filenames (without extension) as keys
val_text_dict = {}

# Load text descriptions from each text file and match them with the images
for text_file in val_text_files:
    val_image_name = os.path.splitext(text_file)[0]
    val_text_file_path = os.path.join(val_text_files_path, text_file)

    with open(val_text_file_path, 'r') as file:
        val_text_descriptions = file.read().splitlines()

    val_text_dict[val_image_name] = val_text_descriptions

    val_text_descriptions = remove_numbers(val_text_descriptions)
    val_text_descriptions = limit_words(val_text_descriptions , word_limit=50)
    val_text_dict[val_image_name] = val_text_descriptions

# Convert image paths to strings by extracting the file name from the full path
val_image_filenames = [os.path.splitext(os.path.basename(val_image_path.filename))[0] for val_image_path in val_dataset["image"]]
val_text_filenames = [os.path.splitext(os.path.basename(val_image_path.filename))[0] + '_desc'  for val_image_path in val_dataset["image"]]
# Add the "text" column to the dimg dataset
val_dataset = val_dataset.add_column("caption", [val_text_dict[filename] for filename in val_text_filenames])
val_dataset = val_dataset.add_column("image_path", val_items_list)
val_dataset = val_dataset.add_column("height", val_heights_list)
val_dataset = val_dataset.add_column("width", val_widths_list)
val_dataset = val_dataset.add_column("image_id", val_image_id_list)
#dataset = dataset.add_column("caption_id", text_id_list)
# Print the resulting dataset with image paths and text descriptions
#text_dict['COCO_train2014_000000000049_desc']
val_dataset = val_dataset.add_column("file_name", val_image_filenames)

# %%
len(dataset["image_id"])

# %%
dataset[5]

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# from datasets import Dataset
# from PIL import Image as PILImage
# import numpy as np
# import pandas as pd

# image_list = []
# text_list = []
# file_name_list = []
# path_list = []
# heights_list = []
# widths_list = []
# image_id_list = []
# text_id_list = []

# image_id_counter = 0
# text_id_counter = 0

# # Iterate over the rows of the original dataset
# for row in dataset:
#     image = row['image']
#     text_descriptions = row['text']
#     file_name = row['file_name']
#     image_path = row['image_path']
#     height = row['height']
#     width = row['width']
#     image_id = row['image_id']

#     image_np = np.array(image)
#     image_bytes = image_np.tobytes()
#     # Count the occurrences of each unique text description
#     unique_texts = set(text_descriptions)
#     for text in unique_texts:
#         count = text_descriptions.count(text)
#         # Duplicate the image, text, and file_name for the number of occurrences
#         for _ in range(count):
#             image_list.append(image_bytes)
#             text_list.append(text)
#             file_name_list.append(file_name)
#             path_list.append(image_path)
#             heights_list.append(height)
#             widths_list.append(width)
#             image_id_list.append(image_id)
#             text_id_list.append(text_id_counter)
#             text_id_counter += 1
# data_dict = {
#     'image_id' : image_id_list,
#     'caption_id': text_id_list,
#     #'image': image_list,
#     'caption': text_list,
#     'height': heights_list,
#     'width': widths_list,
#     'file_name': file_name_list,
#     'coco_url': path_list,
#     'image_path': path_list
# }
# df = pd.DataFrame(data_dict)
# # Create the new dataset with the expanded instances
# new_dataset = Dataset.from_pandas(df)

# # Print the new dataset
# print(new_dataset)


# %%
# #DEPTH
# from datasets import Dataset
# from PIL import Image as PILImage
# import numpy as np
# import pandas as pd

# depth_image_list = []
# depth_text_list = []
# depth_file_name_list = []
# depth_path_list = []
# depth_heights_list = []
# depth_widths_list = []
# depth_image_id_list = []
# depth_text_id_list = []

# depth_image_id_counter = 0
# depth_text_id_counter = 0

# # Iterate over the rows of the original dataset
# for row in depth_dataset:
#     depth_image = row['image']
#     depth_text_descriptions = row['text']
#     depth_file_name = row['file_name']
#     depth_image_path = row['image_path']
#     depth_height = row['height']
#     depth_width = row['width']
#     depth_image_id = row['image_id']

#     image_np = np.array(depth_image)
#     depth_image_bytes = image_np.tobytes()
#     # Count the occurrences of each unique text description
#     unique_texts = set(depth_text_descriptions)
#     for depth_text in unique_texts:
#         count = depth_text_descriptions.count(depth_text)
#         # Duplicate the image, text, and file_name for the number of occurrences
#         for _ in range(count):
#             depth_image_list.append(depth_image_bytes)
#             depth_text_list.append(depth_text)
#             depth_file_name_list.append(depth_file_name)
#             depth_path_list.append(depth_image_path)
#             depth_heights_list.append(depth_height)
#             depth_widths_list.append(depth_width)
#             depth_image_id_list.append(depth_image_id)
#             depth_text_id_list.append(depth_text_id_counter)
#             depth_text_id_counter += 1
# depth_data_dict = {
#     'image_id' : depth_image_id_list,
#     'caption_id': depth_text_id_list,
#     #'image': image_list,
#     'caption': depth_text_list,
#     'height': depth_heights_list,
#     'width': depth_widths_list,
#     'file_name': depth_file_name_list,
#     'coco_url': depth_path_list,
#     'image_path': depth_path_list
# }
# depth_df = pd.DataFrame(depth_data_dict)
# # Create the new dataset with the expanded instances
# depth_dataset = Dataset.from_pandas(depth_df)

# # Print the new dataset
# print(depth_dataset)


# %%
# #VAL_DEPTH
# from datasets import Dataset
# from PIL import Image as PILImage
# import numpy as np
# import pandas as pd

# val_depth_image_list = []
# val_depth_text_list = []
# val_depth_file_name_list = []
# val_depth_path_list = []
# val_depth_heights_list = []
# val_depth_widths_list = []
# val_depth_image_id_list = []
# val_depth_text_id_list = []

# val_depth_image_id_counter = 0
# val_depth_text_id_counter = 0

# # Iterate over the rows of the original dataset
# for row in val_depth_dataset:
#     val_depth_image = row['image']
#     val_depth_text_descriptions = row['text']
#     val_depth_file_name = row['file_name']
#     val_depth_image_path = row['image_path']
#     val_depth_height = row['height']
#     val_depth_width = row['width']
#     val_depth_image_id = row['image_id']

#     image_np = np.array(val_depth_image)
#     val_depth_image_bytes = image_np.tobytes()
#     # Count the occurrences of each unique text description
#     unique_texts = set(val_depth_text_descriptions)
#     for val_depth_text in unique_texts:
#         count = val_depth_text_descriptions.count(val_depth_text)
#         # Duplicate the image, text, and file_name for the number of occurrences
#         for _ in range(count):
#             val_depth_image_list.append(val_depth_image_bytes)
#             val_depth_text_list.append(val_depth_text)
#             val_depth_file_name_list.append(val_depth_file_name)
#             val_depth_path_list.append(val_depth_image_path)
#             val_depth_heights_list.append(val_depth_height)
#             val_depth_widths_list.append(val_depth_width)
#             val_depth_image_id_list.append(val_depth_image_id)
#             val_depth_text_id_list.append(val_depth_text_id_counter)
#             val_depth_text_id_counter += 1
# val_depth_data_dict = {
#     'image_id' : val_depth_image_id_list,
#     'caption_id': val_depth_text_id_list,
#     #'image': image_list,
#     'caption': val_depth_text_list,
#     'height': val_depth_heights_list,
#     'width': val_depth_widths_list,
#     'file_name': val_depth_file_name_list,
#     'coco_url': val_depth_path_list,
#     'image_path': val_depth_path_list
# }
# val_depth_df = pd.DataFrame(val_depth_data_dict)
# # Create the new dataset with the expanded instances
# val_depth_dataset = Dataset.from_pandas(val_depth_df)

# # Print the new dataset
# print(val_depth_dataset)


# %%
# #EVALUATION
# from datasets import Dataset
# from PIL import Image as PILImage
# import numpy as np
# import pandas as pd
# # Assuming you already have the dataset with the format you provided
# val_image_list = []
# val_text_list = []
# val_file_name_list = []
# val_path_list = []
# val_heights_list = []
# val_widths_list = []
# val_image_id_list = []
# val_text_id_list = []

# val_image_id_counter = 0
# val_text_id_counter = 0

# # Iterate over the rows of the original dataset
# for row in val_dataset:
#     image = row['image']
#     text_descriptions = row['text']
#     file_name = row['file_name']
#     image_path = row['image_path']
#     height = row['height']
#     width = row['width']
#     image_id = row['image_id']

#     image_np = np.array(image)
#     image_bytes = image_np.tobytes()
#     # Count the occurrences of each unique text description
#     unique_texts = set(text_descriptions)
#     for text in unique_texts:
#         count = text_descriptions.count(text)
#         # Duplicate the image, text, and file_name for the number of occurrences
#         for _ in range(count):
#             val_image_list.append(image_bytes)
#             val_text_list.append(text)
#             val_file_name_list.append(file_name)
#             val_path_list.append(image_path)
#             val_heights_list.append(height)
#             val_widths_list.append(width)
#             val_image_id_list.append(image_id)
#             val_text_id_list.append(text_id_counter)
#             val_text_id_counter += 1
# val_data_dict = {
#     'image_id' : val_image_id_list,
#     'caption_id': val_text_id_list,
#     #'image': image_list,
#     'caption': val_text_list,
#     'height': val_heights_list,
#     'width': val_widths_list,
#     'file_name': val_file_name_list,
#     'coco_url': val_path_list,
#     'image_path': val_path_list
# }
# val_df = pd.DataFrame(val_data_dict)
# # Create the new dataset with the expanded instances
# new_val_dataset = Dataset.from_pandas(val_df)

# # Print the new dataset
# print(new_val_dataset)


# %%
# #DEPTH
# depth_dataset

# %%
# #VAL_DEPTH
# val_depth_dataset

# %%
#EVALUATION
#new_val_dataset

# %%
#new_dataset

# %%
from PIL import Image

# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                      padding="max_length",
                      max_length=max_target_length).input_ids

    return labels


# image preprocessing step
def feature_extraction_fn(image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """

    if check_image==False:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file)
                #if img.mode != 'RGB':
                   #img = img.convert('RGB')
                images.append(img)
                to_keep.append(True)               
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values
    #     #images = [Image.open(image_file) for image_file in image_paths]
    #     processed_images = []
    #     for image_file in sorted(glob.glob(f"{image_paths}/*.JPG")): #image_paths:
    #         with Image.open(image_file) as img:
    #             processed_images.append(img)
                
    # # Use feature_extractor to obtain encoder inputs
    #     encoder_inputs = feature_extractor(images=processed_images, return_tensors="pt")
    # #encoder_inputs = feature_extractor(images=images, return_tensors="np")

    #return encoder_inputs.pixel_values

def preprocess_fn(examples, max_target_length, check_image = True):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']
    captions = examples['caption']
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)

    return model_inputs

# %%
# depth_images = []
# to_keep = []
# for image_file in depth_list:
#             try:
#                 img = Image.open(image_file)
#                 depth_images.append(img)
#                 to_keep.append(True)
#             except Exception:
#                 to_keep.append(False)
# else:
#     depth_images = [Image.open(image_file) for image_file in depth_list]

# %%
# depth_images[1].getbands()

# %%
# import cv2

# # Load an image from file
# image = cv2.imread(depth_list[1], cv2.IMREAD_GRAYSCALE)

# # Check the pixel value at a specific location (e.g., row 100, column 200)
# pixel_value = image[240, 255]
# print(f'Pixel value at (100, 200): {pixel_value}')

# # Check the shape of the image (height, width, number of channels)
# height, width = image.shape
# print(f'Image shape: Height={height}, Width={width}')


# %%
# for image_file in depth_list:
#     d_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

# %%
dataset[3]

# %%
preprocess_fn(dataset, 15, check_image = True)


# %%
processed_dataset = dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 200},
    #remove_columns=dataset.column_names
)

# %%
# #DEPTH
# preprocess_fn(depth_dataset, 128, check_image = True)

# %%
# #VAL_DEPTH
# preprocess_fn(val_depth_dataset, 128, check_image = True)

# %%
#EVALUATION
preprocess_fn(val_dataset, 15, check_image = True)

# %%
#EVALUATION
val_processed_dataset = val_dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 200},
    #remove_columns=val_dataset.column_names
)

# %%
processed_dataset

# %%
val_processed_dataset

# %%
# #DEPTH
# depth_processed_dataset = depth_dataset.map(
#     function=preprocess_fn,
#     batched=True,
#     fn_kwargs={"max_target_length": 128},
#     #remove_columns=new_dataset.column_names
# )

# %%
# #VAL_DEPTH
# val_depth_processed_dataset = val_depth_dataset.map(
#     function=preprocess_fn,
#     batched=True,
#     fn_kwargs={"max_target_length": 128},
#     #remove_columns=new_dataset.column_names
# )

# %%
#pixel_values1 = np.array(processed_dataset['pixel_values'])
#pixel_values2 = np.array(depth_processed_dataset['pixel_values'])

# %%
#pixel_values2.shape

# %%
#depth_layer = np.zeros((pixel_values2.shape[0], 1, pixel_values2.shape[2], pixel_values2.shape[3]))
#depth_layer

# %%
#pixel_values2_with_depth = np.concatenate([pixel_values2, depth_layer], axis=1)

# %%
#print(pixel_values2_with_depth.shape)

# %%
# depth_values = np.array(depth_processed_dataset['pixel_values'])


# %%
# val_depth_values = np.array(val_depth_processed_dataset['pixel_values'])

# %%
#rbg_values = np.array(processed_dataset['pixel_values'])

# %%
#val_rbg_values = np.array(val_processed_dataset['pixel_values'])

# %%
# depth_values.shape

# %%
#len(rbg_values)

# %%
#rbg_values.shape

# %%
#rbg_values[1][0]

# %%
# new_pixel_values = np.concatenate((rbg_values, depth_values), axis=1)


# %%
# new_val_pixel_values = np.concatenate((val_rbg_values, val_depth_values), axis=1)

# %%
# new_pixel_values = new_pixel_values[:, :4, :, :]
# new_pixel_values.shape

# %%
# new_val_pixel_values = new_val_pixel_values[:, :4, :, :]
# new_val_pixel_values.shape

# %%
# new_pixel_values[2000][2]

# %%
# #processed_dataset['pixel_values'] = pixel_values2_with_depth.tolist()
# import numpy as np
# from datasets import Dataset

# # Assuming you have a dataset named 'processed_dataset' and pixel_values2_with_depth
# # with shape (2944, 4, 224, 224)

# # Convert pixel_values2_with_depth to a list
# #pixel_values2_with_depth_list = pixel_values2_with_depth.tolist()

# # Create a new dataset with the updated 'pixel_values' field
# updated_dataset = Dataset.from_dict({
#     'image_id': processed_dataset['image_id'],  # Include other fields as needed
#     'caption_id': processed_dataset['caption_id'],
#     'caption': processed_dataset['caption'],
#     'height': processed_dataset['height'],
#     'width': processed_dataset['width'],
#     'file_name': processed_dataset['file_name'],
#     'coco_url': processed_dataset['coco_url'],
#     'image_path': processed_dataset['image_path'],
#     'labels': processed_dataset['labels'],
#     'pixel_values': processed_dataset['pixel_values']  # Update pixel_values
# })


# %%
# #VALIDATION
# import numpy as np
# from datasets import Dataset

# # Assuming you have a dataset named 'processed_dataset' and pixel_values2_with_depth
# # with shape (2944, 4, 224, 224)

# # Convert pixel_values2_with_depth to a list
# #pixel_values2_with_depth_list = pixel_values2_with_depth.tolist()

# # Create a new dataset with the updatσσed 'pixel_values' field
# val_updated_dataset = Dataset.from_dict({
#     'image_id': val_processed_dataset['image_id'],  # Include other fields as needed
#     'caption_id': val_processed_dataset['caption_id'],
#     'caption': val_processed_dataset['caption'],
#     'height': val_processed_dataset['height'],
#     'width': val_processed_dataset['width'],
#     'file_name': val_processed_dataset['file_name'],
#     'coco_url': val_processed_dataset['coco_url'],
#     'image_path': val_processed_dataset['image_path'],
#     'labels': val_processed_dataset['labels'],
#     'pixel_values': val_processed_dataset['pixel_values'] # Update pixel_values
# })

# %%
# depth_dataset

# %%
# depth_processed_dataset

# %%
#processed_dataset['pixel_values'][1]

# %%
#depth_processed_dataset['pixel_values'][1]

# %%
#depth_processed_dataset['pixel_values'][1][1][15]

# %%
#depth_processed_dataset['pixel_values'][1][2][15]

# %%
#depth_processed_dataset['pixel_values'][1][0][15]

# %%
#len(depth_processed_dataset['pixel_values'][1][0][223])

# %%
#len(processed_dataset['labels'][2943])

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    output_dir="./image-captioning-output-107epochs",
    num_train_epochs= 50
)

# %%
import evaluate
metric = evaluate.load("rouge")

# %%
import numpy as np

ignore_pad_token_for_loss = True


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# %%
from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset,
    #eval_dataset=val_updated_dataset,
    eval_dataset=val_processed_dataset,
    #train_dataset=processed_dataset['train'],
    #eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)

# %%
trainer.compute_metrics

# %%
# from transformers.trainer_utils import get_last_checkpoint
# import logging
# logger = logging.getLogger()
# if get_last_checkpoint(trainer.args.output_dir) is not None:
#     logger.info("***** continue training *****")
#     last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
#     trainer.train(resume_from_checkpoint=last_checkpoint)
# else:
#     trainer.train()

# %%
#trainer.model

# %%
trainer.train()

# %%
trainer.save_model("./image-captioning-output-107epochs")

# %%
tokenizer.save_pretrained("./image-captioning-output-107epochs")

# %%
from transformers import pipeline
# full dataset trained model can be found at https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
image_captioner = pipeline("image-to-text", model="./image-captioning-output-106epochs", max_new_tokens=10)

# %%
dataset["image"][5]

# %%
image_captioner("test_images/COCO_test2015_000000000014.jpg")

# %%
# x = new_dataset['image_path'][0]
# image = cv2.imread(x)
# height, width, num_channels = image.shape
# image.shape

# %%
# from PIL import Image
# import numpy as np

# # Open an image using Pillow
# image = Image.open(x)

# # Convert the image to RGBA mode (4 channels) and fill the alpha channel with zeros
# if image.mode != 'RGBA':
#     image = image.convert('RGBA')

# # Create an alpha channel with zeros
# alpha = Image.new('L', image.size,255)

# # Add the alpha channel to the image
# image.putalpha(alpha)
# image.save('image_with_alpha.png')


# %%
# from PIL import Image

# # Open the RGB image
# rgb_image = Image.open("your_rgb_image.jpg")

# # Convert the RGB image to RGBA by adding an alpha channel
# rgba_image = rgb_image.convert("RGBA")

# # Set the alpha value for the entire image (255 means fully opaque)
# alpha_value = 255

# # Create a transparent image (fully opaque) with the same size as the original
# alpha_layer = Image.new("L", rgba_image.size, alpha_value)

# # Composite the RGB image and the alpha layer to create an RGBA image
# rgba_image = Image.alpha_composite(rgba_image, alpha_layer)

# # Save the resulting image with the alpha channel
# rgba_image.save("output_image.png")

# %%
# rgb_image = dataset['image'][0]
# rgba_image = Image.new("RGBA", rgb_image.size)

# # Set the alpha value for the entire image (255 means fully opaque)
# alpha_value = 255

# for x in range(rgb_image.width):
#     for y in range(rgb_image.height):
#         r, g, b = rgb_image.getpixel((x, y))
#         rgba_image.putpixel((x, y), (r, g, b, alpha_value))

# # Save the resulting image with the alpha channel
# rgba_image.save("output_image.png")
# z = rgba_image



# %%
# #image_captioner(dataset['image'][1])
# image_captioner(rgba_image)

# %%
dataset['image'][9]

# %%
image_captioner(dataset['image'][9])

# %%
val_processed_dataset

# %%
processed_dataset

# %%
# depth_dataset = load_dataset("csv", data_files="/home/vcl3d/coco_dataset_VOX_mini/train2014_csv/COCO_train2014_000000000009.csv")

# %%
# depth_dataset

# %%
# depth_dataset['train']['depth']

# %%
# depth_dataset['train']['Classes']

# %%
dataset


