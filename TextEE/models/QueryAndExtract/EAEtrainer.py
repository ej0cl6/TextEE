import os, sys, logging, tqdm, pprint
import torch
import numpy as np
from collections import namedtuple
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .EAEmodel import QueryAndExtractEAEModel
from .metadata import Metadata
from .utils import token_to_berttokens, arg_to_token_ids, event_id_to_arg_query_and_mask, from_entity_identifier_to_entity_matrix
from scorer import compute_EAE_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

EAEBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_trigger', 'batch_arguments',
                   'sentence_batch', 'idxs_to_collect', 'is_triggers', 'bert_sentence_lens', 
                   'arg_weight_matrices', 'arg_mapping', 'entity_mapping', 'arg_tags', 
                   'batch_pieces', 'batch_token_lens', 'batch_token_num', 'batch_entities', 'batch_tokens', 
                  ]
EAEBatch = namedtuple('EAEBatch', field_names=EAEBatch_fields, defaults=[None] * len(EAEBatch_fields))

def EAE_collate_fn(batch):
    return EAEBatch(
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
        batch_pieces=[instance["pieces"] for instance in batch], 
        batch_token_lens=[instance["token_lens"] for instance in batch], 
        batch_tokens=[instance["tokens"] for instance in batch], 
        batch_token_num=[instance["token_num"] for instance in batch], 
        batch_entities=[instance["entities"] for instance in batch], 
    )

class QueryAndExtractEAETrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.model = None
        
    @classmethod
    def add_extra_info_fn(cls, instances, raw_data, config):
        extra_info_map = {}
        for dt in raw_data:
            extra_info = {
                "entity_mentions": dt["entity_mentions"] if "entity_mentions" in dt else [], 
            }
            extra_info_map[(dt["doc_id"], dt["wnd_id"])] = extra_info
        for instance in instances:
            instance["extra_info"] = extra_info_map[(instance["doc_id"], instance["wnd_id"])]
        
        return instances
    
    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(os.path.join(checkpoint, "best_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            self.tokenizer = state["tokenizer"]
            self.type_set = state["type_set"]
            self.model = QueryAndExtractEAEModel(self.config, self.tokenizer, self.type_set)
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
            self.model = QueryAndExtractEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)

    def process_data_for_training(self, data):
        assert self.tokenizer and self.model, "Please load model and tokenizer before processing data!"
        logger.info("Processing data...")

        # Prepare argument query meta infos
        if self.config.pretrained_model_name.startswith('bert'):
            sep_tokens = [self.tokenizer.sep_token]
        elif self.config.pretrained_model_name.startswith('roberta'):
            sep_tokens = [self.tokenizer.sep_token, self.tokenizer.sep_token]
        else:
            raise ValueError("not implemented")
        meta_info = self.model.earl_model.meta_info
        arg_tokenizer_ids = arg_to_token_ids(meta_info.arg_set, self.tokenizer)
        arg_to_query_mask_dict = event_id_to_arg_query_and_mask(meta_info.trigger_arg_dic,
                                                                arg_tokenizer_ids,
                                                                meta_info.args_to_ids)

        new_data = []
        for dt in data:
            sent_len = len(dt['tokens'])
            
            arguments = [a for a in dt['arguments']]
            labeled_entities = set([(a[0], a[1]) for a in arguments]) 
            non_labeled_entities = set([(e['start'], e['end'], None, e['text']) for e in dt["extra_info"]["entity_mentions"] if (e['start'], e['end']) not in labeled_entities])
            arguments.extend(list(non_labeled_entities))
            arguments.sort(key=lambda x: (x[0], x[1]))
            
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

            if len(arguments) > self.config.max_entity_count:
                logger.info(f"Too many arguments, cut to {self.config.max_entity_count}.")
                arguments = arguments[:self.config.max_entity_count]
            
            # create entity mapping matrix
            entity_mapping = torch.zeros((sent_len, self.config.max_entity_count))
            for j, x, in enumerate(arguments):
                this_entity_span = x[1] - x[0]
                for pos in range(x[0], x[1]):
                    entity_mapping[pos, j] = 1. / this_entity_span

            # argument annotations
            this_gth_tags_list = []
            for j, x in enumerate(arguments):
                # the corresponding role of a predicted entity (if none, annotate 'O')
                if x[2] != None:
                    this_gth_tags_list.append(meta_info.args_to_ids[x[2]])
                else: # 'O' correspond to None 
                    this_gth_tags_list.append(meta_info.args_to_ids['O'])
                    
            pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            entities = [(e["start"], e["end"], "Entity", e["text"]) for e in dt["extra_info"]["entity_mentions"]]
            
            new_dt = {"doc_id": dt["doc_id"],
                      "wnd_id": dt["wnd_id"],
                      "trigger": dt["trigger"],
                      "arguments": arguments,
                      "sentence": bert_tokens_arg,
                      "idxs_to_collect": first_subword_idxs_arg, 
                      "is_trigger": trigger_indicator, 
                      "bert_sentence_len": bert_sentence_lengths,
                      "arg_weight_matrice": arg_mask,
                      "arg_mapping": arg_type_ids,
                      "entity_mapping": entity_mapping,
                      "arg_tag": this_gth_tags_list,
                      "tokens": dt["tokens"], 
                      "pieces": [p for w in pieces for p in w], 
                      "token_lens": token_lens, 
                      "token_num": len(dt["tokens"]), 
                      "entities": entities,
                     }
            
            new_data.append(new_dt)
        
        logger.info(f"Generate {len(new_data)} QueryAndExtract EAE instances from {len(data)} EAE instances")
        return new_data
    
    def process_data_for_testing_ner(self, data):
        assert self.tokenizer and self.model, "Please load model and tokenizer before processing data!"
        logger.info("Processing data...")

        new_data = []
        for dt in data:
            pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            entities = [(e["start"], e["end"], "Entity", e["text"]) for e in dt["extra_info"]["entity_mentions"]]
            
            new_dt = {"doc_id": dt["doc_id"],
                      "wnd_id": dt["wnd_id"],
                      "trigger": dt["trigger"],
                      "arguments": dt["arguments"],
                      "sentence": [],
                      "idxs_to_collect": [], 
                      "is_trigger": [], 
                      "bert_sentence_len": [],
                      "arg_weight_matrice": [],
                      "arg_mapping": [],
                      "entity_mapping": [],
                      "arg_tag": [],
                      "tokens": dt["tokens"], 
                      "pieces": [p for w in pieces for p in w], 
                      "token_lens": token_lens, 
                      "token_num": len(dt["tokens"]), 
                      "entities": entities,
                     }
            
            new_data.append(new_dt)
        
        logger.info(f"Generate {len(new_data)} QueryAndExtract EAE instances from {len(data)} EAE instances")
        return new_data
    
    def process_data_for_testing(self, data, ner_predictions):
        assert self.tokenizer and self.model, "Please load model and tokenizer before processing data!"
        logger.info("Processing data...")

        # Prepare argument query meta infos
        if self.config.pretrained_model_name.startswith('bert'):
            sep_tokens = [self.tokenizer.sep_token]
        elif self.config.pretrained_model_name.startswith('roberta'):
            sep_tokens = [self.tokenizer.sep_token, self.tokenizer.sep_token]
        else:
            raise ValueError("not implemented")
        meta_info = self.model.earl_model.meta_info
        arg_tokenizer_ids = arg_to_token_ids(meta_info.arg_set, self.tokenizer)
        arg_to_query_mask_dict = event_id_to_arg_query_and_mask(meta_info.trigger_arg_dic,
                                                                arg_tokenizer_ids,
                                                                meta_info.args_to_ids)

        new_data = []
        for dt, ner_pred in zip(data, ner_predictions):
            sent_len = len(dt['tokens'])
            
            assert ((dt["doc_id"], dt["wnd_id"]) + dt["trigger"]) == ((ner_pred["doc_id"], ner_pred["wnd_id"]) + ner_pred["trigger"])
            arguments = [(a[0], a[1], None, "") for a in ner_pred["entities"]]
            arguments.sort(key=lambda x: (x[0], x[1]))
            
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

            if len(arguments) > self.config.max_entity_count:
                logger.info(f"Too many arguments, cut to {self.config.max_entity_count}.")
                arguments = arguments[:self.config.max_entity_count]
            
            # create entity mapping matrix
            entity_mapping = torch.zeros((sent_len, self.config.max_entity_count))
            for j, x, in enumerate(arguments):
                this_entity_span = x[1] - x[0]
                for pos in range(x[0], x[1]):
                    entity_mapping[pos, j] = 1. / this_entity_span

            # argument annotations
            this_gth_tags_list = []
            for j, x in enumerate(arguments):
                # the corresponding role of a predicted entity (if none, annotate 'O')
                if x[2] != None:
                    this_gth_tags_list.append(meta_info.args_to_ids[x[2]])
                else: # 'O' correspond to None 
                    this_gth_tags_list.append(meta_info.args_to_ids['O'])
                    
            pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            entities = [(e["start"], e["end"], "Entity", e["text"]) for e in dt["extra_info"]["entity_mentions"]]
            
            new_dt = {"doc_id": dt["doc_id"],
                      "wnd_id": dt["wnd_id"],
                      "trigger": dt["trigger"],
                      "arguments": arguments,
                      "sentence": bert_tokens_arg,
                      "idxs_to_collect": first_subword_idxs_arg, 
                      "is_trigger": trigger_indicator, 
                      "bert_sentence_len": bert_sentence_lengths,
                      "arg_weight_matrice": arg_mask,
                      "arg_mapping": arg_type_ids,
                      "entity_mapping": entity_mapping,
                      "arg_tag": this_gth_tags_list,
                      "tokens": dt["tokens"], 
                      "pieces": [p for w in pieces for p in w], 
                      "token_lens": token_lens, 
                      "token_num": len(dt["tokens"]), 
                      "entities": entities,
                     }
            
            new_data.append(new_dt)
        
        logger.info(f"Generate {len(new_data)} QueryAndExtract EAE instances from {len(data)} EAE instances")
        return new_data

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data_for_training(train_data)
        internal_dev_data = self.process_data_for_testing_ner(dev_data)
        
        param_optimizer1 = list(self.model.earl_model.bert.named_parameters())
        param_optimizer2 = list(self.model.earl_model.linear.named_parameters())
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
        ner_param_groups = [
            {
                'params': [p for n, p in self.model.ner_model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.ner_base_model_learning_rate, 'weight_decay': self.config.ner_base_model_weight_decay
            },
            {
                'params': [p for n, p in self.model.ner_model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.ner_learning_rate, 'weight_decay': self.config.ner_weight_decay
            },
        ]
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        
        ner_optimizer = AdamW(params=ner_param_groups)
        ner_scheduler = get_linear_schedule_with_warmup(ner_optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.max_epoch)
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
            ner_optimizer.zero_grad()
            cummulate_loss = []
            cummulate_ner_loss = []
            for batch_idx, batch in enumerate(DataLoader(internal_train_data, batch_size=self.config.train_batch_size // self.config.accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=EAE_collate_fn)):
                
                loss, _, ner_loss = self.model(batch)
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss.append(loss.item())
                loss.backward()
                
                ner_loss = ner_loss * (1 / self.config.accumulate_step)
                cummulate_ner_loss.append(ner_loss.item())
                ner_loss.backward()

                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.model.earl_model.parameters(), self.config.grad_clipping)
                    torch.nn.utils.clip_grad_norm_(self.model.ner_model.parameters(), self.config.ner_grad_clipping)
                    optimizer.step()
                    ner_optimizer.step()
                    scheduler.step()
                    ner_scheduler.step()
                    optimizer.zero_grad()
                    ner_optimizer.zero_grad()
                    
            progress.close()
            logger.info(f"Average training loss: {np.mean(cummulate_loss)}")
            logger.info(f"Average training ner_loss: {np.mean(cummulate_ner_loss)}")
            
            # eval dev
            predictions = self.internal_predict(dev_data, split="Dev")
            dev_scores = compute_EAE_scores(predictions, internal_dev_data)

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
        
        
        internal_data1 = self.process_data_for_testing_ner(eval_data)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        ner_predictions = []
        for batch_idx, batch in enumerate(DataLoader(internal_data1, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            batch_pred_entities = self.model.ner_model.predict(batch)
            for doc_id, wnd_id, trigger, pred_entities in zip(batch.batch_doc_id, batch.batch_wnd_id, 
                                                               batch.batch_trigger, batch_pred_entities):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "trigger": trigger, 
                              "entities": pred_entities
                             }
                
                ner_predictions.append(prediction)
        progress.close()
        
        internal_data2 = self.process_data_for_testing(eval_data, ner_predictions)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(internal_data2, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            batch_pred_arguments = self.model.earl_model.predict(batch)
            for doc_id, wnd_id, trigger, pred_arguments in zip(batch.batch_doc_id, batch.batch_wnd_id, 
                                                               batch.batch_trigger, batch_pred_arguments):
                
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "trigger": trigger, 
                              "arguments": [p for p in pred_arguments if p[2] is not None]
                             }
                
                predictions.append(prediction)
        progress.close()
        
        return predictions

    def predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        predictions = self.internal_predict(data, split="Test")
        return predictions