import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForPreTraining
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers.modeling_outputs import Seq2SeqLMOutput
import ipdb

class BartGenEAEModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        
        if self.config.pretrained_model_name.startswith('facebook/bart'):
            self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_name,
                                                          cache_dir=self.config.cache_dir)
            self.model_config.output_attentions = True
            self.model = CopyBart.from_pretrained(self.config.pretrained_model_name,
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
            if num_beams == 1:
                self.model._cache_input_ids = input_ids
            else:
                expanded_return_idx = (
                    torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
                )
                self.model._cache_input_ids = input_ids.index_select(0, expanded_return_idx)
                
            outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
            
        final_output = []
        for bid in range(len(input_ids)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path)

    def load_model(self, load_path):
        self.model.from_pretrained(load_path)

        
# For copy

class CopyBart(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        # If extra model/module, we need to initialize the module here.
        self.linear_copy = nn.Linear(self.config.d_model, 1)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        
        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        assert input_ids.size(0) == outputs.encoder_last_hidden_state.size(0) # batch size
        
        cross_attentions = outputs.cross_attentions 
        cross_attentions = torch.stack(cross_attentions[-1:], dim=1) # TODO: we can change the used layer here.
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate layers
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate heads
        # Now, "cross attentions" is of shape (batch_size, dec_sequence_length, enc_sequence_length)

        # Probability of copying
        p_ori = torch.sigmoid(self.linear_copy(outputs[0]))

        # Merge distribution
        original_word_pro = torch.softmax(lm_logits, dim=-1) * p_ori #[batch, dec_sequence_length, vocab_size]
        copy_words = input_ids.unsqueeze(1).repeat(1, cross_attentions.size(1), 1) #(batch, dec_sequence_length, enc_sequence_length)
        
        input_len = input_ids.size(1)
        lm_logits = torch.scatter_add(original_word_pro, 2, copy_words, cross_attentions[:,:,:input_len]*(1-p_ori))

        eps = 1e-7
        lm_logits = torch.log(lm_logits+eps)
        
        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # loss_fct = CrossEntropyLoss()
            loss_fct = nn.NLLLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
