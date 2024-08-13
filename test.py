import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset
from peft import LoraConfig
from accelerate import Accelerator
import evaluate

from model import VQAModel, Trainer
from data import VQADataset, CollateFn
from utils import Metric


if __name__=='__main__':
    
    log = False
    dev_run = False
    weights_dir = 'VQAModel-step2.pt'
    # language encoder tag
    llm_tag = 'Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa'
    # vision encoder tag
    vit_tag = 'openai/clip-vit-base-patch32'
    
    tokenizer = AutoTokenizer.from_pretrained(llm_tag)
    processor = AutoProcessor.from_pretrained(vit_tag)
    # pads to longest sequence in the batch
    collate_fn = CollateFn(tokenizer)
    
    # load dataset from huggingface hub
    data = load_dataset('bipin/ml_vqa', trust_remote_code=True)
    # testing set
    test_ds = VQADataset(data['test'], processor, split='test')
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    test_size = len(test_ds)
    
    model = VQAModel(llm_tag, vit_tag, tokenizer)

    lora_config = LoraConfig(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha=128,
        lora_dropout=0.5,
        bias='none',
    )
    model.prepare_for_finetuning(lora_config)
    
    # load pre-trained model for evaluation
    if not os.path.exists(weights_dir):
        print(f'[-] No pre-trained weights (`{weights_dir}`) found in the current directory.') 
        print('[+] Downloading pre-trained weights from hub!')
        from huggingface_hub import HfApi
        
        # download pre-trained weights from huggingface hub
        hub = hf = HfApi()
        weights_dir = hf.hf_hub_download(repo_id='bipin/model', filename='VQAModel-step2.pt')
        
    # load the pre-trained weights
    model.load_state_dict(torch.load(weights_dir))
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    opt = None

    accelerator = Accelerator()
    model, test_dl = accelerator.prepare(model, test_dl)
    
    # evaluation metrics
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    metric = Metric([bleu, rouge, meteor], tokenizer)

    trainer = Trainer(
        model, tokenizer, opt, criterion, metric, accelerator,
        log=log, dev_run=dev_run,
    )
    # evaluate the model on test set
    test_loss, test_score = trainer.evaluate(test_dl, test_size, 'Testing', eval_response_only=True)
    
    print(f'\n************************\nEvaluation results\n************************\n test_loss: {test_loss} test_score: {test_score}')