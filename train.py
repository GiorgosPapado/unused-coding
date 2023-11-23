# %%
import os
import datasets
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor,AutoTokenizer
#os.environ["WANDB_DISABLED"] = "true"
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3.0" 


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

from datasets import load_dataset


# %%
# image feature extractor
feature_extractor = AutoImageProcessor.from_pretrained(image_encoder_model)
# text tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_decode_model)

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.encoder.embeddings.patch_embeddings.projection

output_dir = "vit-gpt-model"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

import pandas as pd
from datasets import Dataset
# Load the CSV file into a pandas DataFrame
# df = pd.read_csv('train_dataset.csv')
# dataset_dict = df.to_dict(orient='list')
# dataset = Dataset.from_dict(dataset_dict)

# val_df = pd.read_csv('val_dataset.csv')
# val_dataset_dict = val_df.to_dict(orient='list')
# val_dataset = Dataset.from_dict(val_dataset_dict)
#val_dataset = val_dataset.remove_columns("Unnamed: 0")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# from torch.utils.data import Dataset

# class CustomTensorDataset(Dataset):
#     def __init__(self, file_paths):
#         self.file_paths = file_paths

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         # Load the tensor from file using streaming or any other technique
#         tensor = torch.load(self.file_paths[idx])
#         return tensor
    
# # Define file paths for your training and validation tensors
# train_file_paths = ['full_training_tensor.pt']
# val_file_paths = ['f16val_tensor_data.pt']

# # Create instances of the CustomTensorDataset class
# train_dataset = CustomTensorDataset(train_file_paths)
# val_dataset = CustomTensorDataset(val_file_paths)

# # Use DataLoader for efficient streaming and batching
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


# %%
from tqdm import tqdm
import numpy as np

from PIL import Image
# def feature_extraction_fn(image_file):
# # Define your image_paths and feature_extractor
#     with Image.open(image_file) as img:
#         image_path = feature_extractor(img, return_tensors="np")
#     return image_path.pixel_values  

# def feature_extraction_fn(image_files):
#     # Assuming feature_extractor is defined somewhere
#     # Make sure it can handle batch processing
#     with Image.open(image_files) as img:
#         # If using a Hugging Face feature extractor
#         images = [feature_extractor(img, return_tensors="np")]
#     return images
#     return tokenized_labels
def feature_extraction_fn(image_paths):
    images = []
    for image_file in tqdm(image_paths, desc="Loading Images"):
        try:
            with Image.open(image_file) as file:
                img = file.convert("RGB")
                images.append(img)
        except Exception as e:
            print(f"Error loading image: {image_file}")
            print(f"Error details: {e}")

    if not images:
        print("No valid images found.")
        return None

    # Rest of your code
    encoder_inputs = feature_extractor(images=images, return_tensors="np")
    return encoder_inputs.pixel_values


def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions, 
                      padding="max_length", 
                      max_length=max_target_length).input_ids

    return labels

def preprocess_fn(examples, max_target_length):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']
    captions = examples['caption']
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    #model_inputs['pixel_values'] = feature_extraction_fn(image_paths)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths)#torch.load(tensor)
    return model_inputs

# %%
# def process_and_append(example):
#     # Assuming local_tensor is a PyTorch tensor
#     model_inputs = {}
#     model_inputs['labels'] = tokenization_fn(example['caption'], 80)  
#     model_inputs['pixel_values'] = feature_extraction_fn(example['image_path'])#specific_tensor[image_id] 
#     #inputs = Dataset.from_dict(model_inputs) 
#     return model_inputs

# %%
import pandas as pd

# Replace 'your_dataset.csv' with the actual name of your CSV file
val_file_path = 'val_dataset.csv'
file_path = 'train_dataset.csv'
# Load the dataset from the CSV file into a pandas DataFrame
val_df = pd.read_csv(val_file_path)
df = pd.read_csv(file_path)
# Display the first few rows of the DataFrame to inspect the loaded data
dataset = df.to_dict(orient='records')
val_dataset = val_df.to_dict(orient='records')

# %%
from datasets import Dataset

column_names = ['image', 'caption', 'image_path', 'height', 'width', 'image_id', 'file_name']
columns = {key: [item[key] for item in dataset] for key in column_names}

# Convert the dictionary to a Hugging Face Dataset
processed_dataset = Dataset.from_dict(columns)

# %%
from datasets import Dataset

column_names = ['image', 'caption', 'image_path', 'height', 'width', 'image_id', 'file_name']
columns = {key: [item[key] for item in val_dataset] for key in column_names}

# Convert the dictionary to a Hugging Face Dataset
val_processed_dataset = Dataset.from_dict(columns)

# %%
filtered_dataset = processed_dataset.filter(lambda example: example['caption'] and example['caption'].strip() != ".")

# Print the filtered dataset
print(filtered_dataset)


# %%
val_filtered_dataset = val_processed_dataset.filter(lambda example: example['caption'] and example['caption'].strip() != ".")

# Print the filtered dataset
print(val_filtered_dataset)


# %%
midpoint = len(filtered_dataset) // 2

# Split the dataset into two parts
train_dataset_part1 = filtered_dataset.select(list(range(midpoint)))
train_dataset_part2 = filtered_dataset.select(list(range(midpoint, len(filtered_dataset))))

# %%
mapped_dataset = train_dataset_part2.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 200},
    #remove_columns=processed_dataset.column_names
)

# %%
# import csv
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# #= ['image', 'caption', 'image_path', 'height', 'width', 'image_id', 'file_name']
# def convert_to_csv(record):
#     # Your conversion logic here
#     return {'image_path': record['image_path'], 'caption': record['caption'], 'labels': record['labels'], 'pixel_values': record['pixel_values']}  # Modify this according to your needs

# def convert_dataset_to_csv_parallel(mapped_dataset, csv_file, num_workers=4):
#     with open(csv_file, 'w', newline='') as csvfile:
#         fieldnames = ['image_path', 'caption', 'labels', 'pixel_values']  # Replace with your actual field names
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # Write header
#         writer.writeheader()

#         # Use ProcessPoolExecutor for parallel processing
#         with ProcessPoolExecutor(max_workers=num_workers) as executor:
#             futures = []
            
#             # Iterate over the dataset and submit tasks for parallel processing
#             for record in mapped_dataset:
#                 futures.append(executor.submit(convert_to_csv, record))
            
#             # tqdm is used to display a progress bar
#             for future in tqdm(executor.as_completed(futures), total=len(futures), desc="Converting"):
#                 result = future.result()
#                 writer.writerow(result)

# %%
midpoint = len(val_filtered_dataset) // 2

# Split the dataset into two parts
val_dataset_part1 = val_filtered_dataset.select(list(range(midpoint)))
val_dataset_part2 = val_filtered_dataset.select(list(range(midpoint, len(val_filtered_dataset))))

# %%
val_mapped_dataset = val_dataset_part2.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 200},
    #remove_columns=val_dataset_part1.column_names
)

# %%
# tokenized_captions = []

# # Iterate through each example in val_dataset
# for example in tqdm(dataset, desc="Tokenizing Captions"):
#     # Access the 'caption' field of the example
#     caption = example['caption']
#     # Tokenize the caption and append to the list
#     tokens = tokenization_fn(caption, max_target_length=150)
#     tokenized_captions.append(tokens)

# %%
# for i, entry in enumerate(dataset):
#     # Add the 'labels' column to each entry
#     entry['labels'] = tokenized_captions[i]

# %%
# #VALIDATION
# tokenized_captions = []

# # Iterate through each example in val_dataset
# for example in tqdm(val_dataset, desc="Tokenizing Captions"):
#     # Access the 'caption' field of the example
#     caption = example['caption']
#     # Tokenize the caption and append to the list
#     tokens = tokenization_fn(caption, max_target_length=150)
#     tokenized_captions.append(tokens)

# # Add the 'labels' column to the dataset
# len(tokenized_captions)

# %%
# for i, entry in enumerate(val_dataset):
#     # Add the 'labels' column to each entry
#     entry['labels'] = tokenized_captions[i]

# %%
# from datasets import Dataset

# column_names = ['image', 'caption', 'image_path', 'height', 'width', 'image_id', 'file_name', 'pixel_values', 'labels']
# columns = {key: [item[key] for item in dataset] for key in column_names}

# # Convert the dictionary to a Hugging Face Dataset
# processed_dataset = Dataset.from_dict(columns)

# %%
# from datasets import Dataset

# column_names = ['image', 'caption', 'image_path', 'height', 'width', 'image_id', 'file_name', 'pixel_values', 'labels']
# columns = {key: [item[key] for item in val_dataset] for key in column_names}

# # Convert the dictionary to a Hugging Face Dataset
# val_processed_dataset = Dataset.from_dict(columns)

# %%
# from datasets import Dataset, DatasetDict
# dataset_dict = DatasetDict({
#     "train": processed_dataset,
#     "validation": val_processed_dataset
# })
# dataset_dict

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    output_dir="./image-captioning-output-111epochs",
    resume_from_checkpoint='/data1/ViTgpt2/image-captioning-output-111epochs/checkpoint-8000',
    num_train_epochs= 10
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
# import pandas as pd
# import pyarrow as pa

# # Convert your Hugging Face dataset to a Pandas DataFrame
# df = pd.DataFrame(mapped_dataset)

# # Convert the Pandas DataFrame to a Feather file
# feather_file = 'mapped_train.feather'
# df.to_feather(feather_file)

# %%
# import pandas as pd
# import pyarrow as pa

# # Convert your Hugging Face dataset to a Pandas DataFrame
# val_df = pd.DataFrame(val_mapped_dataset)

# # Convert the Pandas DataFrame to a Feather file
# feather_file = 'mapped_val.feather'
# val_df.to_feather(feather_file)

# %%
from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=mapped_dataset,
    #eval_dataset=val_updated_dataset,
    eval_dataset=val_mapped_dataset,
    #train_dataset=processed_dataset['train'],
    #eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)

# %%
trainer.compute_metrics

# %%
trainer.train()

# %%
trainer.save_model("./image-captioning-output-111epochs")

tokenizer.save_pretrained("./image-captioning-output-111epochs")

# %%
from transformers import pipeline
image_captioner = pipeline("image-to-text", model="./image-captioning-output-111epochs", max_new_tokens=200)

# %%
# caption = "/data1/ViTgpt2/test_images/COCO_test2015_000000000076.jpg"
# image_captioner(caption)

# %%
test_path = "/data1/ViTgpt2/test_images"
filenames = [os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]

# %%
def print_up_to_n_sentences(captions, n):
    for caption in captions:
        generated_text = caption.get('generated_text', '')
        sentences = generated_text.split('.')
        result = '.'.join(sentences[:n])
        #print(result)
    return result

# %%
for filename in filenames:
    generated_captions = image_captioner(filename)
    print_up_to_n_sentences(generated_captions, 4)

# %%
import re
def remove_numbers(text_descriptions):
    clean_text_descriptions = []
    for line in text_descriptions:
        clean_text_descriptions.append((re.sub(r'\d+','', line))[1:])
    return clean_text_descriptions

# %%
text_files_path = "/data1/ViTgpt2/test_images_text"
text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
text_files = sorted(text_files)
all_descriptions = []
#text_files = sorted(text_files[:100])
# Create a dictionary to store text descriptions with image filenames (without extension) as keys

# Load text descriptions from each text file and match them with the images
for text_file in text_files:
    text_file_path = os.path.join(text_files_path, text_file)

    with open(text_file_path, 'r') as file:
        text_descriptions = file.read().splitlines()

    text_descriptions = remove_numbers(text_descriptions)
    all_descriptions.append(text_descriptions)  

# %%
def similarity(sentences_list_1, sentences_list_2):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Function to calculate cosine similarity
    def calculate_similarity(sentence1, sentence2):
        vectorizer = CountVectorizer().fit_transform([sentence1, sentence2])
        vectors = vectorizer.toarray()
        return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # Iterate through sentences and calculate similarity
    for sentence_2 in sentences_list_2[0].split('.'):
        if sentence_2.startswith('The'):
            similarity_scores = []
            for sentence_1 in sentences_list_1:
                if sentence_1.startswith('The'):
                    similarity = calculate_similarity(sentence_1, sentence_2)
                    similarity_scores.append((sentence_1, similarity))

            # Sort similarity scores in descending order
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # Print the top similarity score
            if similarity_scores:
                print(f"For sentence: '{sentence_2}'")
                print(f"Most similar sentence: '{similarity_scores[0][0]}' with similarity score: {similarity_scores[0][1]:.2%}\n") 

# %%
for i, filename in enumerate(filenames):
    generated_captions = image_captioner(filename)
    x = all_descriptions[i]
    y = print_up_to_n_sentences(generated_captions, 3)
    y = y.split('. ')
    print(similarity(x, y))

# %%
y


