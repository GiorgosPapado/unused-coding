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

def feature_extraction_fn(image_files):
    # Assuming feature_extractor is defined somewhere
    # Make sure it can handle batch processing
    # with Image.open(image_files) as img:
    #     # If using a Hugging Face feature extractor
    #     images = [feature_extractor(img, return_tensors="np")]
    # return images
    images = []
    for image_file in image_files:
        with Image.open(image_file) as file:
            images.append(file)
    # Rest of your code
    encoder_inputs = feature_extractor(images=images, return_tensors="np")
    return encoder_inputs.pixel_values
#     return tokenized_labels
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

# def process_and_append(dataset, local_tensor):
#     captions = dataset['caption']
#     dataset['labels'] = tokenization_fn(captions, 200)
#     for example, local_data in zip(dataset, local_tensor):
#         # Assuming local_data is a PyTorch tensor
#         example['pixel_values'] = local_data
#         yield example

# processed_dataset = train_dataset.map(lambda x: process_and_append(x, train_tensor), batched=True)
# val_processed_dataset = val_dataset.map(lambda x: process_and_append(x, val_tensor), batched=True)

# annotations = preprocess_fn(train_dataset, 150, train_tensor)
# train_dataset.add_column("labels", annotations['labels'])
# train_dataset.add_column("pixel_values", annotations['pixel_values'])

# val_annotations = preprocess_fn(val_dataset, 150, val_tensor)
# val_dataset.add_column("labels", val_annotations['labels'])
# val_dataset.add_column("pixel_values", val_annotations['pixel_values'])
# print("s")

# %%
def process_and_append(example):
    # Assuming local_tensor is a PyTorch tensor
    # image_id = example['image_id']
    # specific_tensor = local_tensor[image_id]
    # example['labels'] = tokenization_fn(example['caption'], 80)  # Assuming 'caption' is the key for captions in your dataset
    # example['pixel_values'] = specific_tensor  # Assuming 'image_id' is the key for image IDs in your dataset
    # return example

    #image_id = example['image_id']
    #specific_tensor = local_tensor['image_id']
    model_inputs = {}
    model_inputs['labels'] = tokenization_fn(example['caption'], 80)  
    model_inputs['pixel_values'] = feature_extraction_fn(example['image_path'])#specific_tensor[image_id] 
    #inputs = Dataset.from_dict(model_inputs) 
    return model_inputs

# %%
import pandas as pd

# Replace 'your_dataset.csv' with the actual name of your CSV file
val_file_path = 'val_data.csv'
file_path = 'train_data.csv'
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
# %%
mapped_dataset = val_processed_dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    #remove_columns=ds['train'].column_names
)

# %%
tokenized_captions = []

# Iterate through each example in val_dataset
for example in tqdm(dataset, desc="Tokenizing Captions"):
    # Access the 'caption' field of the example
    caption = example['caption']
    # Tokenize the caption and append to the list
    tokens = tokenization_fn(caption, max_target_length=150)
    tokenized_captions.append(tokens)

# %%
for i, entry in enumerate(dataset):
    # Add the 'labels' column to each entry
    entry['labels'] = tokenized_captions[i]

# %%
#VALIDATION
tokenized_captions = []

# Iterate through each example in val_dataset
for example in tqdm(val_dataset, desc="Tokenizing Captions"):
    # Access the 'caption' field of the example
    caption = example['caption']
    # Tokenize the caption and append to the list
    tokens = tokenization_fn(caption, max_target_length=150)
    tokenized_captions.append(tokens)

# Add the 'labels' column to the dataset
len(tokenized_captions)

# %%
for i, entry in enumerate(val_dataset):
    # Add the 'labels' column to each entry
    entry['labels'] = tokenized_captions[i]

# %%
from datasets import Dataset

column_names = ['image', 'caption', 'image_path', 'height', 'width', 'image_id', 'file_name', 'pixel_values', 'labels']
columns = {key: [item[key] for item in dataset] for key in column_names}

# Convert the dictionary to a Hugging Face Dataset
processed_dataset = Dataset.from_dict(columns)

# %%
from datasets import Dataset

column_names = ['image', 'caption', 'image_path', 'height', 'width', 'image_id', 'file_name', 'pixel_values', 'labels']
columns = {key: [item[key] for item in val_dataset] for key in column_names}

# Convert the dictionary to a Hugging Face Dataset
val_processed_dataset = Dataset.from_dict(columns)

# %%
from datasets import Dataset, DatasetDict
dataset_dict = DatasetDict({
    "train": processed_dataset,
    "validation": val_processed_dataset
})
dataset_dict

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    output_dir="./image-captioning-output",
    num_train_epochs= 20
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
    train_dataset=dataset_dict['train'],
    #eval_dataset=val_updated_dataset,
    eval_dataset=dataset_dict['validation'],
    #train_dataset=processed_dataset['train'],
    #eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)

# %%
trainer.compute_metrics

# %%
trainer.train()

# %%
trainer.save_model("./image-captioning-output-107epochs")

tokenizer.save_pretrained("./image-captioning-output-107epochs")

from transformers import pipeline
image_captioner = pipeline("image-to-text", model="./image-captioning-output-107epochs", max_new_tokens=10)
train_dataset["image"][5]
image_captioner(train_dataset['image'][5])
image_captioner("test_images/COCO_test2015_000000000014.jpg")



