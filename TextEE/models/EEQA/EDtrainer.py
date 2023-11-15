import os, sys, logging, tqdm, pprint
import torch
import numpy as np
from collections import namedtuple
from transformers import RobertaTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .EDmodel import EEQAEDModel
from scorer import compute_ED_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

EDBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_pieces', 'batch_token_lens', 'batch_token_num', 'batch_text', 'batch_triggers']
EDBatch = namedtuple('EDBatch', field_names=EDBatch_fields, defaults=[None] * len(EDBatch_fields))

def ED_collate_fn(batch):
    return EDBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance["tokens"] for instance in batch], 
        batch_pieces=[instance["pieces"] for instance in batch], 
        batch_token_lens=[instance["token_lens"] for instance in batch], 
        batch_token_num=[instance["token_num"] for instance in batch], 
        batch_text=[instance["text"] for instance in batch], 
        batch_triggers=[instance["triggers"] for instance in batch], 
    )

class EEQAEDTrainer(BasicTrainer):
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
            self.model = EEQAEDModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, do_lower_case=False, add_prefix_space=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, do_lower_case=False, use_fast=False)
            self.model = EEQAEDModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)
    
    def process_data(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        logger.info("Removing overlapping triggers and over-length examples")
        
        # greedily remove overlapping triggers
        n_total = 0
        new_data = []
        for dt in data:
            
            n_total += 1
            
            if len(dt["tokens"]) > self.config.max_length:
                continue
            
            no_overlap_flag = np.ones((len(dt["tokens"]), ), dtype=bool)
            new_triggers = []
            for trigger in dt["triggers"]:
                start, end = trigger[0], trigger[1]
                if end - start > 1:
                    continue
                if np.all(no_overlap_flag[start:end]):
                    new_triggers.append(trigger)
                    no_overlap_flag[start:end] = False
            
            pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces] 
            
            new_dt = {"doc_id": dt["doc_id"], 
                      "wnd_id": dt["wnd_id"], 
                      "tokens": dt["tokens"], 
                      "pieces": [p for w in pieces for p in w], 
                      "token_lens": token_lens, 
                      "token_num": len(dt["tokens"]), 
                      "text": dt["text"], 
                      "triggers": new_triggers
                     }
            
            new_data.append(new_dt)
                
        logger.info(f"There are {len(new_data)}/{n_total} ED instances after removing overlapping triggers and over-length examples")

        return new_data

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data(train_data)
        internal_dev_data = self.process_data(dev_data)
        
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.base_model_learning_rate, 'weight_decay': self.config.base_model_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            },
        ]
        
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        optimizer = AdamW(params=param_groups)
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
                
                loss = self.model(batch)
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
            dev_scores = compute_ED_scores(predictions, internal_dev_data, metrics={"trigger_id", "trigger_cls"})

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
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            batch_pred_triggers = self.model.predict(batch)
            for doc_id, wnd_id, tokens, text, pred_triggers in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, 
                                                                   batch.batch_text, batch_pred_triggers):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
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
