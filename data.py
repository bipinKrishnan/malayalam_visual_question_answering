import torch
from torch.utils.data import Dataset


class CollateFn:
    def __init__(self, tokenizer):
        """
        Tokenizes the question and response.
        Collates the images, inputs and response in
        the batch.
        """
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        images, input_text, output_text = [], [], []
        for b1, b2, b3 in batch:
            images.append(b1)
            input_text.append(b2)
            # add <eos> token after output text (answer + rational)
            output_text.append(b3+self.tokenizer.eos_token)
        
        # stack the images in the batch
        images = torch.cat(images, dim=0)
        # tokenize the question
        input_text = self.tokenizer(
            input_text, 
            padding=True, 
            truncation=False,
            return_tensors='pt',
        )
        # input will contain concatenated image repr
        # and question, so add an extra attention mask
        # for the image repr
        input_text['attention_mask'] = torch.cat(
            [
                torch.full((len(batch), 1), 1), 
                input_text['attention_mask']
            ],
            dim=1
        )
        
        # tokenize the response
        output_text = self.tokenizer(
            output_text, 
            padding=True, 
            truncation=False,
            return_tensors='pt',
        )['input_ids']
        
        return images, input_text, output_text


class VQADataset(Dataset):
    def __init__(self, ds, processor, split):
        super().__init__()
        self.ds = ds
        self.processor = processor
        self.split = split
        self.available_splits = ('train', 'val', 'test')
        # template for training set
        self.prompt = """ 
        ### Question:
        {question}

        ### Answer:
        {answer}

        ### Reason:
        {reason}
        """
        # template for validation and test sets
        if self.split in self.available_splits[-2:]:
            self.qs_prompt = """
            ### Question:
            {question}
            
            ### Answer:
            """
            
    def __getitem__(self, idx):
        img = self.processor(images=self.ds[idx]['image'], return_tensors='pt')['pixel_values']
        text = self.prompt.format(
            question=self.ds[idx]['question_ml'],
            answer=self.ds[idx]['answer_ml'],
            reason=self.ds[idx]['reason_ml'],
        )
        
        # for validation and test sets
        if self.split in self.available_splits[-2:]:
            qs = self.qs_prompt.format(
                question=self.ds[idx]['question_ml'],
            )
            return img, qs, text
        return img, text, text
        
    def __len__(self):
        return len(self.ds)