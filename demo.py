import os
import torch
from transformers import AutoTokenizer, AutoProcessor
from peft import LoraConfig
import gradio as gr

from model import VQAModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_dir = 'VQAModel-step2.pt'
# language encoder tag
llm_tag = 'Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa'
# vision encoder tag
vit_tag = 'openai/clip-vit-base-patch32'

tokenizer = AutoTokenizer.from_pretrained(llm_tag)
processor = AutoProcessor.from_pretrained(vit_tag)

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
model = model.to(device)
model.eval()


@torch.no_grad()
def run_inference(image, question, max_tokens):
    # pro-process inputs for the model
    image = processor(images=image, return_tensors='pt')['pixel_values']
    question = f"""
            ### Question:
            {question}
            
            ### Answer:
            """
    question = tokenizer(
        question, 
        padding=False, 
        truncation=False, 
        return_tensors='pt'
    )['input_ids']

    image, question = image.to(device), question.to(device)
    # generate the answer
    output = model.generate(image, question, max_new_tokens=max_tokens)[0]
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    output = output.split('### Answer:')[-1]
    return output


if __name__=='__main__':
    
    # create the gradio interface and launch
    app = gr.Interface(
        fn=run_inference,
        inputs=[
            'image', 
            gr.Textbox(label='Question'), 
            gr.Slider(label='Max tokens to generate', value=50, minimum=1, maximum=70, step=1)
        ],
        outputs=[gr.Textbox(label='Generated answer'),],
    )

    app.launch(share=True)