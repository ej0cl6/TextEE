import torch
import torch.nn as nn
# from transformers import BartForConditionalGeneration
from .prefix_gen_bart import PrefixGenBartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, AutoConfig, AutoModel
from .AMRBART.AMRBartTokenizer import AMRBartTokenizer, AMRRobertaTokenizer
from torch.nn import NLLLoss
from transformers.modeling_outputs import Seq2SeqLMOutput
import ipdb, logging, re

class CopyBartWithReg(PrefixGenBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # If extra model/module, we need to initialize the module here.
        self.linear_copy = nn.Linear(self.config.d_model, 1)

    def forward(
        self,
        input_ids=None,
        prefix=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                print('decoder_input_shifting')
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            prefix=prefix,
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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias # batch x dec_sequence_length x vocab_size

        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == outputs.encoder_last_hidden_state.size(0) # batch size
        except:
            ipdb.set_trace()

        cross_attentions = outputs.cross_attentions 
        # This is in tuple format, and each of them is of shape (batch_size, num_heads, dec_sequence_length, enc_sequence_length).
        # This are the attentions weights of the decoder’s cross-attention layer, after the attention softmax.
        cross_attentions = torch.stack(cross_attentions[-1:], dim=1) # TODO: we can change the used layer here.
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate layers
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate heads
        # Now, "cross attentions" is of shape (batch_size, dec_sequence_length, enc_sequence_length)

        # For cases on using cross_prefix, we need to remove the prefix attention length
        if self.config.use_cross_prefix:
            cross_attentions = cross_attentions[:, :, self.config.prefix_length:]

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
            # loss_fct = CrossEntropyLoss()
            # # masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # masked_lm_loss = loss_fct(torch.flatten(cross_attentions, start_dim=0, end_dim=1), labels.view(-1))
            loss_fct = NLLLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # add regularizer to p_ori
            masked_lm_loss += torch.mean(p_ori)

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

class CopyBart(PrefixGenBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # If extra model/module, we need to initialize the module here.
        self.linear_copy = nn.Linear(self.config.d_model, 1)

    def forward(
        self,
        input_ids=None,
        prefix=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                print('decoder_input_shifting')
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            prefix=prefix,
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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias # batch x dec_sequence_length x vocab_size

        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == outputs.encoder_last_hidden_state.size(0) # batch size
        except:
            ipdb.set_trace()

        cross_attentions = outputs.cross_attentions 
        # This is in tuple format, and each of them is of shape (batch_size, num_heads, dec_sequence_length, enc_sequence_length).
        # This are the attentions weights of the decoder’s cross-attention layer, after the attention softmax.
        cross_attentions = torch.stack(cross_attentions[-1:], dim=1) # TODO: we can change the used layer here.
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate layers
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate heads
        # Now, "cross attentions" is of shape (batch_size, dec_sequence_length, enc_sequence_length)

        # For cases on using cross_prefix, we need to remove the prefix attention length
        if self.config.use_cross_prefix:
            cross_attentions = cross_attentions[:, :, self.config.prefix_length:]

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
            # loss_fct = CrossEntropyLoss()
            # # masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # masked_lm_loss = loss_fct(torch.flatten(cross_attentions, start_dim=0, end_dim=1), labels.view(-1))
            loss_fct = NLLLoss(ignore_index=-100)
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

class PureCopyBart(PrefixGenBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        prefix=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                print('decoder_input_shifting')
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            prefix=prefix,
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
        # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias # batch x dec_sequence_length x vocab_size

        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == outputs.encoder_last_hidden_state.size(0) # batch size
        except:
            ipdb.set_trace()

        cross_attentions = outputs.cross_attentions 
        # This is in tuple format, and each of them is of shape (batch_size, num_heads, dec_sequence_length, enc_sequence_length).
        # This are the attentions weights of the decoder’s cross-attention layer, after the attention softmax.

        # This is for investigating why regularizer works.
        cross_attentions = torch.stack(cross_attentions[-1:], dim=1) # TODO: we can change the used layer here.
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate layers
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate heads
        # Now, "cross attentions" is of shape (batch_size, dec_sequence_length, enc_sequence_length)

        # For cases on using cross_prefix, we need to remove the prefix attention length
        if self.config.use_cross_prefix:
            cross_attentions = cross_attentions[:, :, self.config.prefix_length:]

        copy_words = input_ids.unsqueeze(1).repeat(1, cross_attentions.size(1), 1) #(batch, dec_sequence_length, enc_sequence_length)
        
        lm_logits = torch.scatter_add(outputs[0].new_zeros(outputs[0].size(0), outputs[0].size(1), self.config.vocab_size), 2, copy_words, cross_attentions)

        eps = 1e-7
        lm_logits = torch.log(lm_logits+eps)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = NLLLoss(ignore_index=-100)
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

####
# For AMR Integration
####
class AMRT5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = (T5ForConditionalGeneration.from_pretrained(config.AMR_model_path)).encoder.cuda()
        self.max_graph_len = 512
        # self.max_sent_len  = 90
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def get_encoder_output(self, stripped_graphs):
        # Form encodings and tokenize
        input_text = ['%s' % graph for graph in stripped_graphs]
        input_encodings = self.tokenizer.batch_encode_plus(input_text, padding=True,
                    truncation=True, max_length=self.max_graph_len, return_overflowing_tokens=True)
        # # Check if any graphs were truncated (requires return_overflowing_tokens=True)
        clip = [l > 0 for l in input_encodings['num_truncated_tokens']]
        if any(clip):
            print("overlength")
        # Convert to tensors
        input_ids = torch.LongTensor(input_encodings['input_ids']).cuda()
        attention_mask = torch.LongTensor(input_encodings['attention_mask']).cuda()
        # Get encoder outputs [batch_size, max_graph_length, 768]
        encoder_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return encoder_output['last_hidden_state'], attention_mask

class AMRBart(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config.AMR_model_path)
        self.tokenizer = AMRBartTokenizer.from_pretrained(config.AMR_model_path)
        self.model = BartForConditionalGeneration.from_pretrained(config.AMR_model_path).cuda()
        self.max_graph_len = 512
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.model.encoder

    def get_encoder_output(self, stripped_graphs):
        input_text = ['%s' % graph for graph in stripped_graphs]
        input_encodings = [
            [self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id] + 
            [self.tokenizer.amr_bos_token_id] + self.tokenizer.tokenize_amr(itm.split())[:self.max_graph_len -5] + 
            [self.tokenizer.amr_eos_token_id] for itm in input_text]
        
        # padding
        max_batch_length = max(len(x) for x in input_encodings)
        attention_mask = [[1]*len(x) + [0]*(max_batch_length - len(x)) for x in input_encodings]
        input_ids = [x + [self.tokenizer.pad_token_id]*(max_batch_length - len(x)) for x in input_encodings]
        
        # truncation
        if max_batch_length > self.max_graph_len:
            input_ids = [x[:self.max_graph_len] for x in input_ids]
            attention_mask = [x[:self.max_graph_len] for x in attention_mask]
            print("overlength")

        # Convert to tensors
        input_ids = torch.LongTensor(input_ids).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda()
        # Get encoder outputs [batch_size, max_graph_length, 1024]
        encoder_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return encoder_output['last_hidden_state'], attention_mask

class AMRRoberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config.AMR_model_path)
        self.tokenizer = AMRRobertaTokenizer.from_pretrained(config.AMR_model_path) 
        self.model = AutoModel.from_pretrained(config.AMR_model_path).cuda()
        self.max_graph_len = 512
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_encoder_output(self, stripped_graphs):
        input_text = ['%s' % graph for graph in stripped_graphs]
        input_encodings = [
            [self.tokenizer.bos_token_id] + self.tokenizer.tokenize_amr(itm.split())[:self.max_graph_len -2] + 
            [self.tokenizer.eos_token_id] for itm in input_text]

        # padding
        max_batch_length = max(len(x) for x in input_encodings)
        attention_mask = [[1]*len(x) + [0]*(max_batch_length - len(x)) for x in input_encodings]
        input_ids = [x + [self.tokenizer.pad_token_id]*(max_batch_length - len(x)) for x in input_encodings]
        
        # truncation
        if max_batch_length > self.max_graph_len:
            input_ids = [x[:self.max_graph_len] for x in input_ids]
            attention_mask = [x[:self.max_graph_len] for x in attention_mask]
            print("overlength")

        # Convert to tensors
        input_ids = torch.LongTensor(input_ids).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda()
        # Get encoder outputs [batch_size, max_graph_length, 1024]
        encoder_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return encoder_output['last_hidden_state'], attention_mask