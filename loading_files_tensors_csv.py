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
train_dataset = load_dataset("sadassa17/rgb-spatial-dataset", split='train', streaming=True)
val_dataset = load_dataset("sadassa17/rgb-spatial-dataset", split='validation', streaming=True)

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
val_dataset = val_dataset.remove_columns("Unnamed: 0")
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
# text preprocessing step
#gpt version

train_tensor = torch.load('full_training_tensor.pt')
#val_tensor = torch.load('val_tensor_data.pt')
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    tokenized_labels = []
    # Initialize tqdm progress bar with the total number of captions
    with tqdm(total=len(captions), desc="Tokenizing Captions") as pbar:
        # Iterate through captions and tokenize them
        for caption in captions:
            # Tokenize the caption using the tokenizer
            tokens = tokenizer(caption, padding="max_length", max_length=max_target_length).input_ids
            # Append tokenized caption to the list
            tokenized_labels.append(tokens)
            pbar.update(1)  # Update the progress bar
    # Convert the list of tokenized labels to a PyTorch tensor
    labels = torch.tensor(tokenized_labels, dtype=torch.long)
    pbar.close()

    return labels

def preprocess_fn(examples, max_target_length, tensor):
    """Run tokenization + image feature extraction"""
    #image_paths = examples['image_path']
    captions = examples['caption']
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    #model_inputs['pixel_values'] = feature_extraction_fn(image_paths)
    model_inputs['pixel_values'] = tensor#torch.load(tensor)
    checker = Dataset.from_dict(model_inputs)
    return checker

def process_and_append(dataset, local_tensor):
    for example, local_data in zip(dataset, local_tensor):
        # Assuming local_data is a PyTorch tensor
        example['new_key'] = local_data.tolist()
        yield example

processed_dataset = train_dataset.map(lambda x: process_and_append(x, train_tensor), batched=True)
val_processed_dataset = val_dataset.map(lambda x: process_and_append(x, val_tensor), batched=True)

annotations = preprocess_fn(train_dataset, 150, train_tensor)
train_dataset.add_column("labels", annotations['labels'])
train_dataset.add_column("pixel_values", annotations['pixel_values'])

val_annotations = preprocess_fn(val_dataset, 150, val_tensor)
val_dataset.add_column("labels", val_annotations['labels'])
val_dataset.add_column("pixel_values", val_annotations['pixel_values'])
print("s")
# %%
#y = tokenization_fn(val_dataset["caption"], 100)
# %%
#preprocess_fn(examples=val_dataset, max_target_length=100, tensor=loaded_tensor)


# %%
#tensor = torch.cat((loaded_tensor1, loaded_tensor2, loaded_tensor3), dim=0)

# %%
#print(tensor.size())


# processed_dataset = train_dataset.map(
#     function=preprocess_fn(train_dataset, 150, loaded_tensor1),
#     batched=True
#     #remove_columns=val_dataset.column_names,
# )

# Apply the preprocess_fn to the validation dataset using map
# val_processed_dataset = val_dataset.map(
#     function=preprocess_fn(val_dataset, 150, tensor),
#     batched=True,
#     fn_kwargs={"max_target_length": 200}
#     #remove_columns=val_dataset.column_names,
# )

###original version***    
# val_processed_dataset = val_dataset.map(
#     function=preprocess_fn(val_dataset, max_target_length=200, tensor=loaded_tensor),
#     batched=True,
#     fn_kwargs={"max_target_length": 200},
#     #remove_columns=dataset.column_names
# )
# %%
# preprocess_fn(dataset, 15, check_image = True)

# processed_dataset = dataset.map(
#     function=preprocess_fn,
#     batched=True,
#     fn_kwargs={"max_target_length": 200},
#     #remove_columns=val_dataset.column_names
# )

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    output_dir="./image-captioning-output-107epochs",
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
    train_dataset=processed_dataset,
    #eval_dataset=val_updated_dataset,
    eval_dataset=val_processed_dataset,
    #train_dataset=processed_dataset['train'],
    #eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)

trainer.compute_metrics

trainer.train()

trainer.save_model("./image-captioning-output-107epochs")

tokenizer.save_pretrained("./image-captioning-output-107epochs")

from transformers import pipeline
image_captioner = pipeline("image-to-text", model="./image-captioning-output-107epochs", max_new_tokens=10)
dataset["image"][5]
image_captioner(dataset['image'][5])
image_captioner("test_images/COCO_test2015_000000000014.jpg")




