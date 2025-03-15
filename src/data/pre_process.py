import torch
import pickle
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer,PaliGemmaProcessor
import yaml
import os
from torch.utils.data import Dataset
path = "C:/Users/ASUS/.cache/kagglehub/models/google/paligemma-2/transformers/paligemma2-3b-pt-224/1"
processor = PaliGemmaProcessor.from_pretrained(path,local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_LOC = "flaviagiammarino___vqa-rad"

class VQADataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.max_length = 512
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        image = self.dataset[idx]['image']
        image = image.convert("RGB")

        text = f"<image> answer en {question}"

        encoding = self.processor(
            text=[text],  
            images=[image],  
            suffix=[answer],  
            padding="max_length",  
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"  
        )

        for k, v in encoding.items():
            if k != "input_ids":  
                encoding[k] = v.squeeze().to(torch.bfloat16).to(device)
            else:
                encoding[k] = v.squeeze().long().to(device)

        return encoding

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))["dataset_loc"]
    data = load_dataset(os.path.join(config["raw"], DATA_LOC))
    train_data = data["train"]
    test_data = data["test"]
    print("VQA dataset loaded successfully!!! lenght of train data is ", len(train_data), " and test data is ", len(test_data))
    train_dataset = VQADataset(dataset=train_data,
                          processor=processor)
    test_dataset = VQADataset(dataset=test_data,
                          processor=processor )
    
    os.makedirs(os.path.join(config["preprocess"], DATA_LOC) , exist_ok=True)
    with open(os.path.join(config["preprocess"], DATA_LOC, "train_dataset.pkl"), "wb") as f:
        pickle.dump(train_dataset, f)
    with open(os.path.join(config["preprocess"], DATA_LOC, "test_dataset.pkl"), "wb") as f:
        pickle.dump(test_dataset, f)

    print(f"Processed data , saved to {os.path.join(config['preprocess'], DATA_LOC)}")