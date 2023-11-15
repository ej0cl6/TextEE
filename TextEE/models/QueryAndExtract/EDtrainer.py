import os, sys, logging, tqdm, pprint, json
import torch
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer as spacy_tokenizer
from collections import namedtuple
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .EDmodel import QueryAndExtractEDModel
from .utils import token_to_berttokens, get_pos, trigger_bio_to_ids, convert_trigger_to_bio
from scorer import compute_ED_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

EDBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_event_type', 'batch_trigger_bio', 'batch_pos_tag', 'batch_bert_sent', 'batch_event_idx_to_collect', 'batch_sent_idx_to_collect']
EDBatch = namedtuple('EDBatch', field_names=EDBatch_fields, defaults=[None] * len(EDBatch_fields))

def ED_collate_fn(batch):
    return EDBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance["tokens"] for instance in batch], 
        batch_event_type=[instance["event_type"] for instance in batch], 
        batch_trigger_bio=[instance["trigger_bio"] for instance in batch], 
        batch_pos_tag=[instance["pos_tag"] for instance in batch], 
        batch_bert_sent=[instance["bert_sent"] for instance in batch], 
        batch_event_idx_to_collect=[instance["event_idx_to_collect"] for instance in batch], 
        batch_sent_idx_to_collect=[instance["sent_idx_to_collect"] for instance in batch],
    )

class QueryAndExtractEDTrainer(BasicTrainer):
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
            self.model = QueryAndExtractEDModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            elif self.config.pretrained_model_name.startswith('bert-'):
                self.tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            special_tokens_dict = {'additional_special_tokens': ['<entity>', '</entity>', '<event>', '</event>', '[EVENT]']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model = QueryAndExtractEDModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)

    def process_data(self, data, sampling=False):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        logger.info("Processing data...")
        
        spacy_tagger = spacy.load("en_core_web_lg")
        spacy_tagger.tokenizer = spacy_tokenizer(spacy_tagger.vocab)

        # get metadata
        with open(self.config.pattern_path, 'r') as f:
            raw_template = json.loads(f.read())[self.config.dataset]
        event_types = self.type_set['trigger']

        event_template = {}
        for e in event_types:
            event_template[e] = raw_template[e].replace('[CLS]', self.tokenizer.cls_token)
            if self.config.pretrained_model_name.startswith('roberta'):
                event_template[e] = event_template[e].replace('[SEP]', self.tokenizer.sep_token+'-'+self.tokenizer.sep_token)
        
        event_template_bert = {}
        for e in event_types:
            temp = event_template[e].split('-')
            event_template_bert[e] = token_to_berttokens(temp, self.tokenizer, template=True)

        new_data = []
        for idx, dt in enumerate(tqdm.tqdm(data, ncols=100)):
            tokens = dt['tokens']
            # ignore long sentences
            # if len(tokens) > 256:
            #     continue
            
            bert_tokens, to_collect = token_to_berttokens(tokens, self.tokenizer, False)
            pos_tag = get_pos(tokens, spacy_tagger)
            event_bio = [convert_trigger_to_bio(trigger, len(tokens)) for trigger in dt['triggers']]
            trigger_bio_dic = trigger_bio_to_ids(event_bio, event_types, len(tokens))

            # randomly select neg data 
            if sampling == True:
                pos_event = [trigger[2] for trigger in dt['triggers']]
                neg_event = [event_type for event_type in event_types if event_type not in pos_event]
                np.random.shuffle(neg_event)
                selected_events = set(pos_event + neg_event[:int(self.config.sampling * len(neg_event))])
            else:
                selected_events = event_types

            for event_type in selected_events:
                this_template = event_template[event_type].split('-')
                this_template_bert, this_template_to_collect = event_template_bert[event_type]
                this_tokens = this_template + tokens + [self.tokenizer.sep_token]

                this_trigger_bio = trigger_bio_dic[event_type]
                bert_sent = this_template_bert + bert_tokens[:] + [self.tokenizer.sep_token]

                sent_idx_to_collect = [0 for _ in range(len(this_template_bert))] + to_collect[:] + [0]
                # data_tuple = (this_tokens, event_type, this_trigger_bio,
                #             None, pos_tag, bert_sent, this_template_to_collect, sent_idx_to_collect)
                
                new_dt = {
                    "doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "tokens": this_tokens,
                    "event_type": event_type,
                    "trigger_bio": this_trigger_bio,
                    "pos_tag": pos_tag,
                    "bert_sent": bert_sent,
                    "event_idx_to_collect": this_template_to_collect,
                    "sent_idx_to_collect": sent_idx_to_collect
                }
                new_data.append(new_dt)
                
        logger.info(f"Generate {len(new_data)} QAE ED instances from {len(data)} ED instances")

        return new_data
    

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data(train_data, sampling=True)
        internal_dev_data = self.process_data(dev_data)
        
        # optimizer
        param_optimizer1 = list(self.model.bert.named_parameters())
        param_optimizer1 = [n for n in param_optimizer1 if 'pooler' not in n[0]]
        param_optimizer2 = list(self.model.linear.named_parameters())
        param_optimizer2.append(('W', self.model.W))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_groups = [
            {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
            'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)],
            'weight_decay': self.config.weight_decay, 'lr': self.config.learning_rate},
            {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        optimizer = AdamW(params=param_groups, lr=self.config.base_model_learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.max_epoch)
        
        best_scores = {"trigger_cls": {"f1": 0.0}}
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
                                                         shuffle=True, drop_last=False, collate_fn=ED_collate_fn)):
                _, _, loss = self.model(batch)
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
            # Note: changed from internal_dev_data to dev_data
            dev_scores = compute_ED_scores(predictions, dev_data, metrics={"trigger_id", "trigger_cls"})

            # print scores
            print(f"Dev {epoch}")
            print_scores(dev_scores)
            
            if dev_scores["trigger_cls"]["f1"] >= best_scores["trigger_cls"]["f1"]:
                logger.info("Saving best model")
                state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                best_scores = dev_scores
                best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))
        
        
    def internal_predict(self, eval_data, split="Dev"):
        # one batch predicts one sample, batch_size = event_count
        eval_batch_num = len(eval_data) // self.model.event_count
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.model.event_count, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            pred_triggers = self.model.predict(batch)
            prediction = {
                "doc_id": batch.batch_doc_id[0],  
                "wnd_id": batch.batch_wnd_id[0], 
                "triggers": pred_triggers
            }
                
            predictions.append(prediction)
        progress.close()
        
        return predictions

    
    def predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        internal_data = self.process_data(data)
        predictions = self.internal_predict(internal_data, split="Test")
        return predictions