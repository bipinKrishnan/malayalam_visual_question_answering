import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from accelerate import Accelerator
import wandb
import evaluate
from tqdm.auto import tqdm

from model import VQAModel, Trainer
from data import VQADataset, CollateFn
from utils import seed_everything, Metric

import warnings
warnings.filterwarnings('ignore')
        

if __name__=='__main__':
    
    seed_everything(seed=3)
    
    # hyper-parameters
    bs = 16
    epochs = 3
    lr = 2e-4
    weight_decay = 0.01
    
    log = False
    dev_run = False
    
    # language encoder tag
    llm_tag = 'Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa'
    # vision encoder tag
    vit_tag = 'openai/clip-vit-base-patch32'
    
    if log:
        run = wandb.init(
            project='Visual Question Answering',
            name='Pre-training',
            reinit=True,
        )
        
        run.log({'Learning rate': lr})
        run.log({'Epochs': epochs})
        run.log({'Batch size': bs})
    
    # load dataset from huggingface hub
    data = load_dataset('bipin/ml_vqa', trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(llm_tag)
    # make pad token is same as <eos>
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # pre-processing for vision model
    processor = AutoProcessor.from_pretrained(vit_tag)
    
    # evaluation metrics
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    metric = Metric([bleu, rouge, meteor], tokenizer)
    
    # pads to longest sequence in the batch
    collate_fn = CollateFn(tokenizer)

    # training set
    train_ds = VQADataset(data['train'], processor, split='train')
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    train_size = len(train_ds)

    # validation set
    val_ds = VQADataset(data['validation'], processor, split='val')
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    val_size = len(val_ds)
    
    # initialize the model
    model = VQAModel(llm_tag, vit_tag, tokenizer)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    accelerator = Accelerator()
    model, opt, train_dl, val_dl = accelerator.prepare(model, opt, train_dl, val_dl)

    trainer = Trainer(
        model, tokenizer, opt, criterion, metric, accelerator, 
        log=log, dev_run=dev_run,
    )
    
    for epoch in tqdm(range(epochs), desc='Epoch'):
        train_loss, train_score = trainer.train(train_dl, train_size)
        val_loss, val_score = trainer.evaluate(val_dl, val_size, 'Validation')

        print(
            f'\n************************\nEpoch: {epoch+1}\n************************\n train_loss: {train_loss} train_score: {train_score} \n\n val_loss: {val_loss} val_score: {val_score}'
        )
        trainer.save_model('VQAModel-step1.pt')

    if log:
        wandb.finish()
    


