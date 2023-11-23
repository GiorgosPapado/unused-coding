

# %%
import torch

# %%
import onnxruntime as rt
from pathlib import Path
import timeit
from copy import deepcopy
from onnxruntime import InferenceSession
from onnxruntime.transformers.optimizer import optimize_model
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# %%
def benchmark(f, name=""):
    # warmup
    for _ in range(10):
        f()
    seconds_per_iter = timeit.timeit(f, number=100) / 100
    print(
        f"{name}:",
        f"{seconds_per_iter * 1000:.3f} ms",
    )


# %%
model_base_id = "madlag/bert-large-uncased-whole-word-masking-finetuned-squadv2"
model_pruned_id = "madlag/bert-large-uncased-wwm-squadv2-x2.63-f82.6-d16-hybrid-v1"
device = "cuda"

# %%
cuda_available = torch.cuda.is_available()
cudnn_available = torch.backends.cudnn.enabled
print(cuda_available)
print(cudnn_available)

# %%
import onnxruntime
onnxruntime.get_available_providers()

# %%
model_path = Path("models/bert")
tokenizer_base = AutoTokenizer.from_pretrained(model_base_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_base_id).to(device)
model.save_pretrained(model_path)

# %%
model_pruned_path = Path("models/bert_pruned")
tokenizer_pruned = AutoTokenizer.from_pretrained(model_pruned_id)
model_pruned = AutoModelForQuestionAnswering.from_pretrained(model_pruned_id).to(device)
model_pruned.save_pretrained(model_pruned_path)

# %%
model_onnx_path = Path("models/bert_onnx_pruned")
model_onnx = ORTModelForQuestionAnswering.from_pretrained(model_pruned_id, export=True, provider="CUDAExecutionProvider")
model_onnx.save_pretrained(model_onnx_path)

# %%
optimized_onnx_path = str(model_onnx_path / "optimized.onnx")
optimized_model = optimize_model(input=str(model_onnx_path / "model.onnx"), model_type="bert", use_gpu=True)
optimized_model.save_model_to_file(optimized_onnx_path)

# %%
optimized_fp16_model_path = str(model_onnx_path / "optimized_fp16.onnx")
optimized_fp16_model = deepcopy(optimized_model)
optimized_fp16_model.convert_float_to_float16()
optimized_fp16_model.save_model_to_file(optimized_fp16_model_path)

# %%
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs_base = tokenizer_base(question, text, return_tensors="pt").to(device)
inputs = tokenizer_pruned(question, text, return_tensors="pt").to(device)
inputs_onnx = dict(tokenizer_pruned(question, text, return_tensors="np"))


# %%
import onnxruntime as ort
ort.get_device()

# %%
ort.get_available_providers()

# %%
providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"]
sess = InferenceSession(str(model_onnx_path / "model.onnx"), providers=providers)
optimized_sess = InferenceSession(str(model_onnx_path / "optimized.onnx"), providers=providers)
optimized_fp16_sess = InferenceSession(
    str(model_onnx_path / "optimized_fp16.onnx"), providers=providers
)

# %%
# %% Test inference times for all variants
benchmark(lambda: model(**inputs_base), "Pytorch")
benchmark(lambda: model_pruned(**inputs), "Pruned Pytorch")
benchmark(lambda: sess.run(None, input_feed=inputs_onnx), "Pruned ONNX")
benchmark(lambda: optimized_sess.run(None, input_feed=inputs_onnx), "Pruned ONNX optimized")
benchmark(lambda: optimized_fp16_sess.run(None, input_feed=inputs_onnx), "Pruned ONNX optimized fp16")

# %%
MAX_SEQUENCE_LENGTH=512
for n in [1, 4, 64, 256, 512]:
    print(f"====== Tokens {n} ======")
    txt = " ".join(["word"] * n)

    pt_inputs_base = tokenizer_base(question, txt, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt").to(device)
    pt_inputs = tokenizer_pruned(question, txt, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt").to(device)
    ort_inputs = dict(tokenizer_pruned(question, txt, max_length=MAX_SEQUENCE_LENGTH, return_tensors="np"))

    benchmark(lambda: model(**pt_inputs), f"Pytorch ({n} tokens)")
    benchmark(lambda: model_pruned(**pt_inputs), f"Pruned Pytorch ({n} tokens)")
    benchmark(lambda: sess.run(None, ort_inputs), f"Pruned ONNX ({n} tokens)")
    benchmark(lambda: optimized_sess.run(None, ort_inputs), f"Pruned ONNX optimized ({n} tokens)")
    benchmark(
        lambda: optimized_fp16_sess.run(None, ort_inputs),
        f"Pruned ONNX optimized fp16 ({n} tokens)",
    )
    
# %%
from pathlib import Path
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from evaluate import evaluator
from transformers import QuestionAnsweringPipeline
import pandas as pd

# %%
# Load the dataset for evaluation
data = load_dataset("squad_v2", split="validation").shuffle(seed=123).select(range(10000))
task_evaluator = evaluator("question-answering")
results = []
index = ["Base model", "Pruned", "ONNX", "ONNX Optimized", "ONNX Optimized FP16"]

# %%
def evaluate_model(model, tokenizer):
  eval_results = task_evaluator.compute(
      model_or_pipeline=model,
      tokenizer=tokenizer,
      data=data,
      metric="squad_v2",
      squad_v2_format=True
  )
  return eval_results

# %%
# model_path = Path("models/bert/")
# device = "cuda"
# tokenizer = AutoTokenizer.from_pretrained("madlag/bert-large-uncased-wwm-squadv2-x2.63-f82.6-d16-hybrid-v1")
# model = ORTModelForQuestionAnswering.from_pretrained(model_path, file_name="model.onnx", provider='CUDAExecutionProvider').to(device)
# pipeline = QuestionAnsweringPipeline(model, tokenizer, task="question-answering")
# results.append(evaluate_model(pipeline, tokenizer))

# %%
model_path = Path("models/bert")
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("madlag/bert-large-uncased-wwm-squadv2-x2.63-f82.6-d16-hybrid-v1")
model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
results.append(evaluate_model(model, tokenizer))

# %%
# model_pruned_path = Path("models/bert_pruned/")
# model_pruned = ORTModelForQuestionAnswering.from_pretrained(model_pruned_path, file_name="model.onnx", provider='TensorrtExecutionProvider').to(device)
# pipeline = QuestionAnsweringPipeline(model, tokenizer, task="question-answering")
# results.append(evaluate_model(pipeline, tokenizer))

# %%
model_pruned_path = Path("models/bert_pruned")
model_pruned = AutoModelForQuestionAnswering.from_pretrained(model_pruned_path).to(device)
results.append(evaluate_model(model_pruned, tokenizer))

# %%
model_files = ["model.onnx", "optimized.onnx", "optimized_fp16.onnx"]

for file_name in model_files:
  onnx_model = ORTModelForQuestionAnswering.from_pretrained("models/bert_onnx_pruned/", file_name=file_name, provider='CUDAExecutionProvider').to(device)
  pipeline = QuestionAnsweringPipeline(onnx_model, tokenizer, task="question-answering")
  results.append(evaluate_model(pipeline, tokenizer))


# %%
results_df = pd.DataFrame(results, index=index)
results_df[["best_exact", "best_f1", "total_time_in_seconds", "latency_in_seconds"]]


