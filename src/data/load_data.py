from datasets import load_dataset
import yaml 
import os 
config_yaml = yaml.safe_load(open("config.yaml", "r"))
PATH = os.path.join(config_yaml["dataset_loc"]["raw"])
ds = load_dataset("flaviagiammarino/vqa-rad" , cache_dir = PATH)
print(ds)