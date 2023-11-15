import os, sys, logging, tqdm, pprint
import torch
import numpy as np
from collections import namedtuple
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .EARLmodel import QueryAndExtractEARLModel
from .metadata import Metadata
from .utils import token_to_berttokens, arg_to_token_ids, event_id_to_arg_query_and_mask, from_entity_identifier_to_entity_matrix
from scorer import compute_EARL_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

EARLBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_trigger', 'batch_arguments',
                    'sentence_batch', 'idxs_to_collect', 'is_triggers', 'bert_sentence_lens', 
                    'arg_weight_matrices', 'arg_mapping', 'entity_mapping', 'arg_tags']
EARLBatch = namedtuple('EARLBatch', field_names=EARLBatch_fields, defaults=[None] * len(EARLBatch_fields))

def EARL_collate_fn(batch):
    return EARLBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_trigger=[instance["trigger"] for instance in batch], 
        batch_arguments=[instance["arguments"] for instance in batch], 
        sentence_batch=[instance["sentence"] for instance in batch], 
        idxs_to_collect=[instance["idxs_to_collect"] for instance in batch], 
        is_triggers=[instance["is_trigger"] for instance in batch], 
        bert_sentence_lens=[instance["bert_sentence_len"] for instance in batch], 
        arg_weight_matrices=[instance["arg_weight_matrice"] for instance in batch], 
        arg_mapping=[instance["arg_mapping"] for instance in batch], 
        entity_mapping=[instance["entity_mapping"] for instance in batch], 
        arg_tags=[torch.tensor(instance["arg_tag"]) for instance in batch], 
    )

class QueryAndExtractEARLTrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.model = None
    
    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(os.path.join(checkpoint, "best_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            self.tokenizer = state["tokenizer"]
            self.type_set = state["type_set"]
            self.model = QueryAndExtractEARLModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            elif self.config.pretrained_model_name.startswith('bert-'):
                self.tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model_name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            special_tokens_dict = {'additional_special_tokens': ['<entity>', '</entity>', '<event>', '</event>', '[EVENT]']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model = QueryAndExtractEARLModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)

    def process_data(self, data):
        assert self.tokenizer and self.model, "Please load model and tokenizer before processing data!"
        logger.info("Processing data...")

        # Prepare argument query meta infos
        if self.config.pretrained_model_name.startswith('bert'):
            sep_tokens = [self.tokenizer.sep_token]
        elif self.config.pretrained_model_name.startswith('roberta'):
            sep_tokens = [self.tokenizer.sep_token, self.tokenizer.sep_token]
        else:
            raise ValueError("not implemented")
        meta_info = self.model.meta_info
        arg_tokenizer_ids = arg_to_token_ids(meta_info.arg_set, self.tokenizer)
        arg_to_query_mask_dict = event_id_to_arg_query_and_mask(meta_info.trigger_arg_dic,
                                                                arg_tokenizer_ids,
                                                                meta_info.args_to_ids)

        new_data = []
        for dt in data:
            sent_len = len(dt['tokens'])
            arg_types = sorted(meta_info.trigger_arg_dic[dt['trigger'][2]]) # prepare arg types for detected trigger
            input_text = [self.tokenizer.cls_token] + dt['tokens'] + sep_tokens + arg_types + ['O', self.tokenizer.sep_token] # input context
            bert_tokens_arg, to_collect_arg = token_to_berttokens(input_text, self.tokenizer)
            first_sep_id = bert_tokens_arg.index(self.tokenizer.sep_token)
            to_collect_arg = to_collect_arg[:first_sep_id]
            to_collect_arg = [i for i in range(len(to_collect_arg)) if to_collect_arg[i]>0] + [first_sep_id, first_sep_id]
            first_subword_idxs_arg = to_collect_arg
            bert_tokens_arg = self.tokenizer.convert_tokens_to_ids(bert_tokens_arg)
            trigger_indicator = [0 for _ in range(sent_len)]
            for _ in range(dt['trigger'][0],dt['trigger'][1]):
                trigger_indicator[_] = 1
            bert_sentence_lengths = len(bert_tokens_arg)
            this_trigger_type_id = meta_info.triggers_to_ids[dt['trigger'][2]]
            arg_mask = arg_to_query_mask_dict[this_trigger_type_id][1]
            arg_type_ids = arg_to_query_mask_dict[this_trigger_type_id][2]

            max_entity_count = 40
            if len(dt['arguments']) > max_entity_count:
                logger.info(f"Too many arguments, cut to {max_entity_count}.")
                dt['arguments'] = dt['arguments'][:max_entity_count]
            
            # create entity mapping matrix
            entity_mapping = torch.zeros((sent_len, max_entity_count))
            for j, x, in enumerate(dt['arguments']):
                this_entity_span = x[1] - x[0]
                for pos in range(x[0], x[1]):
                    entity_mapping[pos, j] = 1. / this_entity_span

            # argument annotations
            this_gth_tags_list = []
            for j, x in enumerate(dt['arguments']):
                # the corresponding role of a predicted entity (if none, annotate 'O')
                if x[2] != None:
                    this_gth_tags_list.append(meta_info.args_to_ids[x[2]])
                else: # 'O' correspond to None 
                    this_gth_tags_list.append(meta_info.args_to_ids['O'])

            new_dt = {"doc_id": dt["doc_id"],
                      "wnd_id": dt["wnd_id"],
                      "trigger": dt["trigger"],
                      "arguments": dt["arguments"],
                      "sentence": bert_tokens_arg,
                      "idxs_to_collect": first_subword_idxs_arg, 
                      "is_trigger": trigger_indicator, 
                      "bert_sentence_len": bert_sentence_lengths,
                      "arg_weight_matrice": arg_mask,
                      "arg_mapping": arg_type_ids,
                      "entity_mapping": entity_mapping,
                      "arg_tag": this_gth_tags_list
                     }
            
            new_data.append(new_dt)
        
        logger.info(f"Generate {len(new_data)} QueryAndExtract EARL instances from {len(data)} EARL instances")
        return new_data

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data(train_data)
        internal_dev_data = self.process_data(dev_data)
        
        param_optimizer1 = list(self.model.bert.named_parameters())
        param_optimizer2 = list(self.model.linear.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_groups = [
            {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
            'weight_decay': self.config.base_model_weight_decay},
            {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)],
            'weight_decay': self.config.weight_decay, 'lr': self.config.learning_rate},
            {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': self.config.learning_rate},
        ]
        
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        optimizer = AdamW(params=param_groups, lr=self.config.base_model_learning_rate, weight_decay=0)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.max_epoch)
        
        best_scores = {"argument_cls": {"f1": 0.0}}
        best_epoch = -1
        
        for epoch in range(1, self.config.max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")
            
            # training step
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train {}'.format(epoch))
            
            self.model.train()
            optimizer.zero_grad()
            cummulate_loss = []
            for batch_idx, batch in enumerate(DataLoader(internal_train_data, batch_size=self.config.train_batch_size // self.config.accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=EARL_collate_fn)):
                
                loss, _ = self.model(batch)
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss.append(loss.item())
                loss.backward()

                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
            progress.close()
            logger.info(f"Average training loss: {np.mean(cummulate_loss)}")
            
            # eval dev
            predictions = self.internal_predict(internal_dev_data, split="Dev")
            dev_scores = compute_EARL_scores(predictions, internal_dev_data)

            # print scores
            print(f"Dev {epoch}")
            print_scores(dev_scores)
            
            if dev_scores["argument_cls"]["f1"] >= best_scores["argument_cls"]["f1"]:
                logger.info("Saving best model")
                state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                best_scores = dev_scores
                best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))
        
    def internal_predict(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=EARL_collate_fn)):
            progress.update(1)
            batch_pred_arguments = self.model.predict(batch)
            for doc_id, wnd_id, trigger, pred_arguments in zip(batch.batch_doc_id, batch.batch_wnd_id, 
                                                               batch.batch_trigger, batch_pred_arguments):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "trigger": trigger, 
                              "arguments": pred_arguments
                             }
                
                predictions.append(prediction)
        progress.close()
        
        return predictions

    def predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        internal_data = self.process_data(data)
        predictions = self.internal_predict(internal_data, split="Test")
        return predictions