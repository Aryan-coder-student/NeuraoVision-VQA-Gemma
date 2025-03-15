# load_dotenv()
# os.getenv("KAGGLE_KEY")
# os.getenv("KAGGLE_USERNAME")
# config_yaml = yaml.safe_load(open("config.yaml", "r"))
# path = kagglehub.model_download("google/paligemma-2/transformers/paligemma2-3b-pt-224")
import os
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers import BitsAndBytesConfig

USE_LORA = False
USE_QLORA = True  
FREEZE_VISION = True

model_id = "C:/Users/ASUS/.cache/kagglehub/models/google/paligemma-2/transformers/paligemma2-3b-pt-224/1"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, use_lora=True, use_qlora=True, freeze_vision=True):
    print("Path to model files:", model_path)

    
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )

    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config if use_qlora else None,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
    )
    
    
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    
    if use_lora:
        lora_config = LoraConfig(
            r=8,  
            lora_alpha=16,  
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # Target modules
            lora_dropout=0.05,  
            bias="none",  
            task_type="CAUSAL_LM"  
        )
        model = get_peft_model(model, lora_config)


    if freeze_vision:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

    
    model = model.to(device)
    model.config.use_cache = False
    # if use_lora or use_qlora:
    #     model.print_trainable_parameters()

    print("Model loaded successfully!")
    return model
# model = load_model(model_id, use_lora=USE_LORA, use_qlora=USE_QLORA, freeze_vision=FREEZE_VISION)



