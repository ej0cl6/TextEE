import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForPreTraining
import ipdb

class XGearEAEModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        
        if self.config.pretrained_model_name.startswith('facebook/bart'):
            self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_name,
                                                          cache_dir=self.config.cache_dir)
            self.model = AutoModelForPreTraining.from_pretrained(self.config.pretrained_model_name,
                                                        cache_dir=self.config.cache_dir, config=self.model_config)
        else:
            raise ValueError("Not implemented.")
            
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def process_data(self, batch):
        # encoder inputs
        inputs = self.tokenizer(batch.batch_input, return_tensors='pt', padding=True)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(batch.batch_target, return_tensors='pt', padding=True)
        batch_size = enc_idxs.size(0)
        
        if self.config.pretrained_model_name.startswith('facebook/bart'):
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.eos_token_id
            # for BART, the decoder input should be:
            # PAD => BOS
            # BOS => A
            # A => B          
        else:
            # t5 case
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.pad_token_id
            # for t5, the decoder input should be:
            # PAD => A
            # A => B
        
        dec_idxs = torch.cat((padding, targets['input_ids']), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
        # dec_idxs = targets['input_ids']
        # dec_idxs[:, 0] = self.tokenizer.eos_token_id
        # dec_attn = targets['attention_mask']
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs

    def forward(self, batch):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        outputs = self.model(input_ids=enc_idxs, 
                             attention_mask=enc_attn, 
                             decoder_input_ids=dec_idxs, 
                             decoder_attention_mask=dec_attn, 
                             labels=lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=4, max_length=50):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        return self.generate(enc_idxs, enc_attn, num_beams, max_length)
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=50, **kwargs):
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
        final_output = []
        for bid in range(len(input_ids)):
            # if self.config.model_name.startswith('google/t5') or self.config.model_name.startswith('t5'):
            #     output_sentence = t5_decode(self.tokenizer, outputs[bid])
            # else:
            #     output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path)

    def load_model(self, load_path):
        self.model.from_pretrained(load_path)
