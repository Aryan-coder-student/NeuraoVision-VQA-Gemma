import sys
import os
import pickle
import yaml 
import torch
torch.cuda.empty_cache()
from transformers import Trainer, TrainingArguments , DataCollatorWithPadding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pre_process import VQADataset
from src.model import load_model 
import warnings
warnings.simplefilter("ignore")
USE_LORA = True
USE_QLORA = True  
FREEZE_VISION = True
model_id = "C:/Users/ASUS/.cache/kagglehub/models/google/paligemma-2/transformers/paligemma2-3b-pt-224/1"
DATA_LOC = "flaviagiammarino___vqa-rad"

pali_model = load_model(model_id, use_lora=USE_LORA, use_qlora=USE_QLORA, freeze_vision=FREEZE_VISION)
config = yaml.safe_load(open("config.yaml", "r"))["dataset_loc"]


with open(os.path.join(config["preprocess"], DATA_LOC, "train_dataset.pkl"), "rb") as f:
    train_dataset = pickle.load(f)
with open(os.path.join(config["preprocess"], DATA_LOC, "test_dataset.pkl"),'rb') as f:
    test_dataset = pickle.load(f)

print("Data loaded successfully!!!")
training_args = TrainingArguments(
            num_train_epochs=3,
            remove_unused_columns=False,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            push_to_hub=False,
            output_dir="paligemma_vqav2",
            bf16=False,
            fp16=True,
            label_names=["labels"],
            dataloader_pin_memory=False,
            
        )



trainer = Trainer(
    model=pali_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()


pali_model.save_pretrained("./paligemma-finetuned")