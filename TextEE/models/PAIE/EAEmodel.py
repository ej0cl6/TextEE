import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartModel
from transformers import BartConfig
from .utils import hungarian_matcher, get_best_span, get_best_span_simple, get_best_indexes, EXTERNAL_TOKENS
from .data import MultiargProcessor
import copy, logging
import ipdb

logger = logging.getLogger(__name__)

def _read_prompt_group(prompt_path):
    with open(prompt_path) as f:
        lines = f.readlines()
    prompts = dict()
    for line in lines:
        if not line:
            continue
        event_type, prompt = line.split(":")
        prompts[event_type] = prompt
    return prompts

class PAIEEAEModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        self.data_processor = MultiargProcessor(config, tokenizer)
        
        if self.config.pretrained_model_name.startswith('facebook/bart'):
            self.model_config = BartConfig.from_pretrained(self.config.pretrained_model_name,
                                                          cache_dir=self.config.cache_dir)
            self.model = BartModel.from_pretrained(self.config.pretrained_model_name,
                                                        cache_dir=self.config.cache_dir)
        else:
            raise ValueError("Not implemented.")

        # Add trigger special tokens and continuous token (maybe in prompt)
        new_token_list = copy.deepcopy(EXTERNAL_TOKENS)
        self.prompts = _read_prompt_group(config.prompt_path)
        for event_type, prompt in self.prompts.items():
            token_list = prompt.split()
            for token in token_list:
                if token.startswith('<') and token.endswith('>') and token not in new_token_list:
                    new_token_list.append(token)
        self.tokenizer.add_tokens(new_token_list)   
        logger.info("Add tokens: {}".format(new_token_list)) 
        self.model.resize_token_embeddings(len(tokenizer))
        self.w_prompt_start = nn.Parameter(torch.rand(self.model_config.d_model, ))
        self.w_prompt_end = nn.Parameter(torch.rand(self.model_config.d_model, ))

        self.model._init_weights(self.w_prompt_start)
        self.model._init_weights(self.w_prompt_end)
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

    def process_data(self, EAEbatch):
        features, batch = self.data_processor.generate_batch(EAEbatch)
        return features, batch

    def forward(self, batch):
        """
        Args:
            multi args post calculation
        """
        # process data
        features, (enc_input_ids, enc_mask_ids, dec_prompt_ids, dec_prompt_mask_ids, target_info, old_tok_to_new_tok_indexs, arg_joint_prompts, arg_list) = self.process_data(batch)

        if self.config.context_representation == 'decoder':
            context_outputs = self.model(
                enc_input_ids,
                attention_mask=enc_mask_ids,
                return_dict=True,
            )
            decoder_context = context_outputs.encoder_last_hidden_state
            context_outputs = context_outputs.last_hidden_state
        else:
            context_outputs = self.model.encoder(
                enc_input_ids,
                attention_mask=enc_mask_ids,
            )
            context_outputs = context_outputs.last_hidden_state
            decoder_context = context_outputs

        decoder_prompt_outputs = self.model.decoder(
                input_ids=dec_prompt_ids,
                attention_mask=dec_prompt_mask_ids,
                encoder_hidden_states=decoder_context,
                encoder_attention_mask=enc_mask_ids,
        )
        decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state   #[bs, prompt_len, H]

        logit_lists = list()
        total_loss = 0.
        for i, (context_output, decoder_prompt_output, arg_joint_prompt, old_tok_to_new_tok_index) in \
            enumerate(zip(context_outputs, decoder_prompt_outputs, arg_joint_prompts, old_tok_to_new_tok_indexs)):
            
            batch_loss = list()
            cnt = 0
            
            output = dict()
            for arg_role in arg_joint_prompt.keys():
                """
                "arg_role": {"tok_s": , "tok_e": }
                """
                prompt_slots = arg_joint_prompt[arg_role]

                start_logits_list = list()
                end_logits_list = list()
                for (p_start,p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                    prompt_query_sub = decoder_prompt_output[p_start:p_end]
                    prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                    
                    start_query = (prompt_query_sub*self.w_prompt_start).unsqueeze(-1) # [1, H, 1]
                    end_query = (prompt_query_sub*self.w_prompt_end).unsqueeze(-1)     # [1, H, 1]

                    start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()  
                    end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()
                    
                    start_logits_list.append(start_logits)
                    end_logits_list.append(end_logits)
                    
                output[arg_role] = [start_logits_list, end_logits_list]

                if self.training:
                    # calculate loss
                    target = target_info[i][arg_role] # "arg_role": {"text": ,"span_s": ,"span_e": }
                    predicted_spans = list()
                    for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                        if self.config.matching_method_train == 'accurate':
                            predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index, self.config.max_span_length))
                        elif self.config.matching_method_train == 'max':
                            predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                        else:
                            raise AssertionError()

                    target_spans = [[s,e] for (s,e) in zip(target["span_s"], target["span_e"])]
                    if len(target_spans)<len(predicted_spans):
                        # need to consider whether to make more 
                        pad_len = len(predicted_spans) - len(target_spans)
                        target_spans = target_spans + [[0,0]] * pad_len
                        target["span_s"] = target["span_s"] + [0] * pad_len
                        target["span_e"] = target["span_e"] + [0] * pad_len
                        
                    if self.config.bipartite:
                        idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                    else:
                        idx_preds = list(range(len(predicted_spans)))
                        idx_targets = list(range(len(target_spans)))
                        if len(idx_targets) > len(idx_preds):
                            idx_targets = idx_targets[0:len(idx_preds)]
                        idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                        idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                    cnt += len(idx_preds)
                    start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds], torch.LongTensor(target["span_s"]).cuda()[idx_targets])
                    end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds], torch.LongTensor(target["span_e"]).cuda()[idx_targets])
                    batch_loss.append((start_loss + end_loss)/2) 
                
            logit_lists.append(output)
            if self.training: # inside batch mean loss
                total_loss = total_loss + torch.sum(torch.stack(batch_loss))/cnt
            
        if self.training:
            return total_loss/len(context_outputs), logit_lists
        else:
            return features, logit_lists

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # build and clean records
            feature_id_list = []
            role_list = []
            full_start_logit_list = []
            full_end_logit_list = []

            # process data
            features, outputs_list = self.forward(batch)

            bs = len(batch[0])
            for i in range(bs):
                predictions = outputs_list[i]
                feature_id = features[i].feature_id
                for arg_role in features[i].arg_list:
                    [start_logits_list, end_logits_list] = predictions[arg_role]
                    for (start_logit, end_logit) in zip(start_logits_list, end_logits_list):
                        feature_id_list.append(feature_id)
                        role_list.append(arg_role)
                        full_start_logit_list.append(start_logit)
                        full_end_logit_list.append(end_logit)
            
            for feature in features:
                feature.init_pred()
                feature.set_gt(self.config.dataset)

            pred_list, cal_time, mask_time, score_time = get_best_indexes(features, feature_id_list, \
                                                                          full_start_logit_list, full_end_logit_list, self.config)

            for (pred, feature_id, role) in zip(pred_list, feature_id_list, role_list):
                pred_span = (pred[0].item(), pred[1].item())
                feature = features[feature_id]
                feature.add_pred(role, pred_span, self.config.dataset)
            
            batch_pred_arguments = []
            for feature in features:
                pred_feature_list = []
                for role in feature.arg_list:
                    pred_role_list = list(set(feature.pred_dict_word[role])) if role in feature.pred_dict_word else list()
                    pred_feature_list.extend([(r[0], r[1], role) for r in pred_role_list])
                batch_pred_arguments.append(pred_feature_list)
        self.train()
        return batch_pred_arguments
