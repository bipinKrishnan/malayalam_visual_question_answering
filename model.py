import torch
from torch import nn
from transformers import AutoTokenizer, AutoProcessor, CLIPVisionModel
from peft import AutoPeftModelForCausalLM, LoraModel

from tqdm.auto import tqdm


class VQAModel(nn.Module):
    def __init__(self, llm_tag, vit_tag, tokenizer, vit_lin_proj=2048):
        super().__init__()
        
        self.tokenizer = tokenizer
        ## language encoder
        self.lang_model = AutoPeftModelForCausalLM.from_pretrained(llm_tag, load_in_4bit=False)
        self.lang_model.requires_grad_(False)
        self.lang_emb = self.lang_model.get_input_embeddings()
        self.lang_emb.requires_grad_(False)
        
        ## image encoder
        self.vision_model = CLIPVisionModel.from_pretrained(vit_tag)
        self.vision_model.requires_grad_(False)
        self.linear_proj = nn.Sequential(
            nn.Linear(768, vit_lin_proj),
            nn.GELU(),
            nn.Linear(vit_lin_proj, vit_lin_proj),
        )
        
    def forward(self, image, question, answers=None):
        img_emb = self.vision_model(pixel_values=image).pooler_output
        # make dim correct before concat with text_emb
        img_emb = self.linear_proj(img_emb)[:, None, :]
        text_emb = self.lang_emb(question['input_ids'])
        
        input_emb = torch.cat([img_emb, text_emb], dim=1)
        output = self.lang_model(
            inputs_embeds=input_emb, 
            attention_mask=question['attention_mask'],
            labels=answers
        )
        return output
    
    def generate(self, image, question, max_new_tokens):
        img_emb = self.vision_model(pixel_values=image).pooler_output
        img_emb = self.linear_proj(img_emb)[:, None, :]
        text_emb = self.lang_emb(question)
        input_emb = torch.cat([img_emb, text_emb], dim=1)
        res = question

        for i in range(max_new_tokens):
            input_emb = input_emb[:, -self.tokenizer.model_max_length:, :]
            logits = self.lang_model(inputs_embeds=input_emb).logits
            pred = logits[:, -1, :].argmax(dim=-1)[:, None]
            res = torch.cat([res, pred], dim=1)

            pred_emb = self.lang_emb(pred)
            input_emb = torch.cat([input_emb, pred_emb], dim=1)
        return res, logits
    
    def prepare_for_finetuning(self, lora_config):
        self.lang_emb.requires_grad_(False)
        self.lang_emb.requires_grad_(False)
        self.lang_model.requires_grad_(True)
        self.lang_model = LoraModel(self.lang_model, config=lora_config, adapter_name='default')
        
        
class Trainer:
    def __init__(
        self, 
        model, 
        tokenizer, 
        opt, 
        criterion, 
        metric, 
        acc_obj, 
        log=False, 
        dev_run=False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.opt = opt
        self.criterion = criterion
        self.metric = metric
        self.log = log
        self.dev_run = dev_run
        self.accelerator = acc_obj
        
    def train(self, train_dl, train_size):
        self.model.train()
        final_loss = 0.0
        
        for img, qs, ans in tqdm(train_dl, desc='Train', leave=False):
            self.opt.zero_grad()
            logits = self.model(img, qs).logits
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]), 
                ans.reshape(-1),
            )
            self.accelerator.backward(loss)
            self.opt.step()

            final_loss += loss
            if self.log:
                run.log({'Train loss per batch': loss})
            _ = self.metric.add_batch(logits.argmax(dim=-1), ans)
            if self.dev_run:
                break
                
        final_loss = (final_loss / train_size).item()
        final_metric = self.metric.compute()
        if self.log:
            run.log({'Train loss': final_loss})
            run.log({f"Bleu (train)": final_metric['bleu']['bleu']})
        return final_loss, final_metric
                
    @torch.no_grad()
    def evaluate(self, eval_dl, eval_size, eval_type, eval_response_only=True):
        self.model.eval()
        final_loss = 0.0
        if self.log: 
            preds_table = wandb.Table(columns=['question', 'answer', 'predicted_answer'])
        
        for img, qs, ans in tqdm(eval_dl, desc=eval_type, leave=False):
            ans_start_idx = qs['input_ids'].shape[-1]
            preds, logits = self.model.generate(
                img, 
                qs['input_ids'], 
                max_new_tokens=(ans.shape[-1]-ans_start_idx),
            )
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]), 
                ans.reshape(-1),
            )

            final_loss += loss
            dec_preds, dec_ans = self.metric.add_batch(preds, ans, ans_only=eval_response_only)
            if self.log:
                run.log({f'{eval_type} loss per batch': loss})
                dec_qs = self.tokenizer.batch_decode(qs['input_ids'], skip_special_tokens=True)
                preds_table.add_data(dec_qs, dec_ans, dec_preds)
            if self.dev_run:
                break
                
        final_loss = (final_loss / eval_size).item()
        final_metric = self.metric.compute()
        if self.log:
            run.log({f'{eval_type} loss': final_loss})
            run.log({f"Bleu ({eval_type.lower()})": final_metric['bleu']['bleu']})        
            run.log({f"{eval_type} data prediction": preds_table})
            
        return final_loss, final_metric
        
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        if self.log:
            wandb_artifact = wandb.Artifact("model", type="model")
            wandb_artifact.add_file(model_path)
            run.log_artifact(wandb_artifact)