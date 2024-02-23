import os, sys, logging, tqdm, pprint
import torch
import numpy as np
from collections import namedtuple, defaultdict
from transformers import RobertaTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .EDmodel import TIModel, ETRModel, ETCModel
from .pattern import event_type_tags
from scorer import compute_ED_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

EDBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_pieces', 'batch_token_lens', 'batch_token_num', 'batch_text', 'batch_triggers', 'batch_spans', 'batch_etc_text', 'batch_etc_label']
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
        batch_spans=[instance["spans"] for instance in batch], 
        batch_etc_text=[instance["etc_text"] for instance in batch], 
        batch_etc_label=[instance["etc_label"] for instance in batch], 
    )


TEMPLATE = "{sentence} {sep} {trigger} {sep} {definition}"

class CEDAREDTrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.model = None
    
    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            ti_state = torch.load(os.path.join(checkpoint, "best_ti_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            etr_state = torch.load(os.path.join(checkpoint, "best_etr_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            etc_state = torch.load(os.path.join(checkpoint, "best_etc_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            self.tokenizer = ti_state["tokenizer"]
            self.type_set = ti_state["type_set"]
            self.ti_model = TIModel(self.config, self.tokenizer)
            self.etr_model = ETRModel(self.config, self.tokenizer, self.type_set)
            self.etc_model = ETCModel(self.config, self.tokenizer)
            self.ti_model.load_state_dict(ti_state['ti_model'])
            self.etr_model.load_state_dict(etr_state['etr_model'])
            self.etc_model.load_state_dict(etc_state['etc_model'])
            self.ti_model.cuda(device=self.config.gpu_device)
            self.etr_model.cuda(device=self.config.gpu_device)
            self.etc_model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, do_lower_case=False, add_prefix_space=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, do_lower_case=False, use_fast=False)
            special_tokens = ["[EVENT]", "[SENT]"]
            logger.info(f"Add tokens {special_tokens}")
            self.tokenizer.add_tokens(special_tokens)
            self.ti_model = TIModel(self.config, self.tokenizer)
            self.etr_model = ETRModel(self.config, self.tokenizer, self.type_set)
            self.etc_model = ETCModel(self.config, self.tokenizer)
    
    def process_data_for_ti(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        logger.info("Removing over-length examples")
        
        # greedily remove overlapping triggers
        n_total = 0
        new_data = []
        for dt in data:
            
            n_total += 1
            
            if len(dt["tokens"]) > self.config.max_length:
                continue
            
            no_overlap_flag = np.ones((len(dt["tokens"]), ), dtype=bool)
            spans = []
            for trigger in dt["triggers"]:
                start, end = trigger[0], trigger[1]
                if np.all(no_overlap_flag[start:end]):
                    spans.append((trigger[0], trigger[1], "Span"))
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
                      "triggers": dt["triggers"],
                      "spans": spans,
                      "etc_text": "",
                      "etc_label": 0,
                     }
            
            new_data.append(new_dt)
                
        logger.info(f"There are {len(new_data)}/{n_total} TI instances after removing over-length examples")

        return new_data
    
    def process_data_for_etc(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        logger.info("Processing ETC data")
        sep = self.tokenizer.sep_token
        
        new_data = []
        for dt in data:
            
            trigger_match = defaultdict(set)
            trigger_ex_all = set()
            for trigger in dt["triggers"]:
                trigger_ex_all.add(trigger[2])
                trigger_match[(trigger[0], trigger[1])].add(trigger[2])

            for trigger in dt["triggers"]:
                definition = event_type_tags[self.config.dataset][trigger[2]]
                sentence = " ".join(dt["tokens"])
                trigger_span = " ".join(dt["tokens"][trigger[0]:trigger[1]])
                text = TEMPLATE.format(definition=definition, sentence=sentence, trigger=trigger_span, sep=sep)
                new_dt = {"doc_id": dt["doc_id"], 
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "pieces": dt["pieces"], 
                          "token_lens": dt["token_lens"], 
                          "token_num": dt["token_num"], 
                          "text": dt["text"], 
                          "triggers": dt["triggers"],
                          "spans": dt["spans"],
                          "etc_text": text,
                          "etc_label": 1,
                         }
                new_data.append(new_dt)
                
                # simple negative
                choices = list(self.type_set["trigger"] - trigger_ex_all)
                np.random.shuffle(choices)
                for neg_type in choices[:self.config.etc_n_neg]:
                    neg_definition = event_type_tags[self.config.dataset][neg_type]
                    text = TEMPLATE.format(definition=neg_definition, sentence=sentence, trigger=trigger_span, sep=sep)
                    new_dt = {"doc_id": dt["doc_id"], 
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "pieces": dt["pieces"], 
                          "token_lens": dt["token_lens"], 
                          "token_num": dt["token_num"], 
                          "text": dt["text"], 
                          "triggers": dt["triggers"],
                          "spans": dt["spans"],
                          "etc_text": text,
                          "etc_label": 0,
                         }
                    new_data.append(new_dt)
                    
                # hard negative
                for neg_type in list(trigger_ex_all - trigger_match[(trigger[0], trigger[1])]):
                    neg_definition = event_type_tags[self.config.dataset][neg_type]
                    text = TEMPLATE.format(definition=neg_definition, sentence=sentence, trigger=trigger_span, sep=sep)
                    new_dt = {"doc_id": dt["doc_id"], 
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "pieces": dt["pieces"], 
                          "token_lens": dt["token_lens"], 
                          "token_num": dt["token_num"], 
                          "text": dt["text"], 
                          "triggers": dt["triggers"],
                          "spans": dt["spans"],
                          "etc_text": text,
                          "etc_label": 0,
                         }
                    new_data.append(new_dt)
                    
        logger.info(f"There are {len(new_data)} ETC instances")

        return new_data

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        
        self.ti_model.cuda(device=self.config.gpu_device)
        internal_train_data = self.process_data_for_ti(train_data)
        internal_dev_data = self.process_data_for_ti(dev_data)
        
        # train TI
        train_batch_num = len(internal_train_data) // self.config.ti_train_batch_size + (len(internal_train_data) % self.config.ti_train_batch_size != 0)
        ti_param_groups = [
            {
                'params': [p for n, p in self.ti_model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.ti_base_model_learning_rate, 'weight_decay': self.config.ti_base_model_weight_decay
            },
            {
                'params': [p for n, p in self.ti_model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.ti_learning_rate, 'weight_decay': self.config.ti_weight_decay
            },
        ]
        ti_optimizer = AdamW(params=ti_param_groups)
        ti_scheduler = get_linear_schedule_with_warmup(ti_optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.ti_warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.ti_max_epoch)
        
        ti_best_scores = {"trigger_id": {"f1": 0.0}}
        ti_best_epoch = -1
        for epoch in range(1, self.config.ti_max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")
            
            # training step
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train TI {}'.format(epoch))
            
            self.ti_model.train()
            ti_optimizer.zero_grad()
            cummulate_ti_loss = []
            for batch_idx, batch in enumerate(DataLoader(internal_train_data, batch_size=self.config.ti_train_batch_size // self.config.ti_accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=ED_collate_fn)):
                
                ti_loss = self.ti_model(batch)
                ti_loss = ti_loss * (1 / self.config.ti_accumulate_step)
                cummulate_ti_loss.append(ti_loss.item())
                ti_loss.backward()

                if (batch_idx + 1) % self.config.ti_accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.ti_model.parameters(), self.config.ti_grad_clipping)
                    ti_optimizer.step()
                    ti_scheduler.step()
                    ti_optimizer.zero_grad()
                    
            progress.close()
            logger.info(f"Average training TI loss: {np.mean(cummulate_ti_loss)}")
            
            # eval dev
            ti_predictions = self.internal_predict_for_ti(internal_dev_data, split="Dev TI")
            dev_scores = compute_ED_scores(ti_predictions, internal_dev_data, metrics={"trigger_id", "trigger_cls"})

            # print scores
            print(f"Dev {epoch}")
            print_scores(dev_scores)
            
            if dev_scores["trigger_id"]["f1"] >= ti_best_scores["trigger_id"]["f1"]:
                logger.info("Saving best TI model")
                ti_state = dict(ti_model=self.ti_model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(ti_state, os.path.join(self.config.output_dir, "best_ti_model.state"))
                ti_best_scores = dev_scores
                ti_best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"ti_best_epoch": ti_best_epoch, "ti_best_scores": ti_best_scores}))
            
        del self.ti_model
        torch.cuda.empty_cache()
            
        # train ETR
        self.etr_model.cuda(device=self.config.gpu_device)
        internal_etr_train_data = [d for d in internal_train_data if len(d["triggers"]) > 0]
        internal_etr_dev_data = [d for d in internal_dev_data if len(d["triggers"]) > 0]
        
        train_batch_num = len(internal_etr_train_data) // self.config.etr_train_batch_size + (len(internal_etr_train_data) % self.config.etr_train_batch_size != 0)
        etr_param_groups = [
            {
                'params': [p for n, p in self.etr_model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.etr_base_model_learning_rate, 'weight_decay': self.config.etr_base_model_weight_decay
            },
            {
                'params': [p for n, p in self.etr_model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.etr_learning_rate, 'weight_decay': self.config.etr_weight_decay
            },
        ]
        etr_optimizer = AdamW(params=etr_param_groups)
        etr_scheduler = get_linear_schedule_with_warmup(etr_optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.etr_warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.etr_max_epoch)
        
        etr_best_scores = 0.0
        etr_best_epoch = -1
        for epoch in range(1, self.config.etr_max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")
            
            # training step
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train ETR {}'.format(epoch))
            
            self.etr_model.train()
            etr_optimizer.zero_grad()
            cummulate_etr_loss = []
            for batch_idx, batch in enumerate(DataLoader(internal_etr_train_data, batch_size=self.config.etr_train_batch_size // self.config.etr_accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=ED_collate_fn)):
                
                etr_loss = self.etr_model(batch)
                etr_loss = etr_loss * (1 / self.config.etr_accumulate_step)
                cummulate_etr_loss.append(etr_loss.item())
                etr_loss.backward()

                if (batch_idx + 1) % self.config.etr_accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.etr_model.parameters(), self.config.etr_grad_clipping)
                    etr_optimizer.step()
                    etr_scheduler.step()
                    etr_optimizer.zero_grad()
                    
            progress.close()
            logger.info(f"Average training ETR loss: {np.mean(cummulate_etr_loss)}")
            
            # eval dev
            etr_predictions = self.internal_predict_for_etr(internal_etr_dev_data, split="Dev ETR")
            dev_scores = self.compute_etr_score(etr_predictions, internal_etr_dev_data)

            # print scores
            print(f"Dev {epoch}")
            print("--------------------------")
            print(f"Hit@{self.config.etr_max_select}: {dev_scores:5.2f}")
            print("--------------------------")
        
            
            if dev_scores >= etr_best_scores:
                logger.info("Saving best ETR model")
                etr_state = dict(etr_model=self.etr_model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(etr_state, os.path.join(self.config.output_dir, "best_etr_model.state"))
                etr_best_scores = dev_scores
                etr_best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"etr_best_epoch": etr_best_epoch, "etr_best_scores": etr_best_scores}))
            
        del self.etr_model
        torch.cuda.empty_cache()
            
        # train ETC
        self.etc_model.cuda(device=self.config.gpu_device)
        internal_etc_train_data = self.process_data_for_etc(internal_etr_train_data)
        internal_etc_dev_data = self.process_data_for_etc(internal_etr_dev_data)
        
        train_batch_num = len(internal_etc_train_data) // self.config.etc_train_batch_size + (len(internal_etc_train_data) % self.config.etc_train_batch_size != 0)
        etc_param_groups = [
            {
                'params': [p for n, p in self.etc_model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.etc_base_model_learning_rate, 'weight_decay': self.config.etc_base_model_weight_decay
            },
            {
                'params': [p for n, p in self.etc_model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.etc_learning_rate, 'weight_decay': self.config.etc_weight_decay
            },
        ]
        etc_optimizer = AdamW(params=etc_param_groups)
        etc_scheduler = get_linear_schedule_with_warmup(etc_optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.etc_warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.etc_max_epoch)
        
        etc_best_scores = 0.0
        etc_best_epoch = -1
        for epoch in range(1, self.config.etc_max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")
            
            # training step
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train ETC {}'.format(epoch))
            
            self.etc_model.train()
            etc_optimizer.zero_grad()
            cummulate_etc_loss = []
            for batch_idx, batch in enumerate(DataLoader(internal_etc_train_data, batch_size=self.config.etc_train_batch_size // self.config.etc_accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=ED_collate_fn)):
                
                etc_loss = self.etc_model(batch)
                etc_loss = etc_loss * (1 / self.config.etc_accumulate_step)
                cummulate_etc_loss.append(etc_loss.item())
                etc_loss.backward()

                if (batch_idx + 1) % self.config.etc_accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.etc_model.parameters(), self.config.etc_grad_clipping)
                    etc_optimizer.step()
                    etc_scheduler.step()
                    etc_optimizer.zero_grad()
                    
            progress.close()
            logger.info(f"Average training ETC loss: {np.mean(cummulate_etc_loss)}")
            
            # eval dev
            etc_predictions = self.internal_predict_for_etc(internal_etc_dev_data, split="Dev ETC")
            dev_scores = self.compute_etc_score(etc_predictions, internal_etc_dev_data)

            # print scores
            print(f"Dev {epoch}")
            print("--------------------------")
            print(f"ACC: {dev_scores:5.2f}")
            print("--------------------------")
            
            if dev_scores >= etc_best_scores:
                logger.info("Saving best ETC model")
                etc_state = dict(etc_model=self.etc_model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(etc_state, os.path.join(self.config.output_dir, "best_etc_model.state"))
                etc_best_scores = dev_scores
                etc_best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"etc_best_epoch": etc_best_epoch, "etc_best_scores": etc_best_scores}))
            
        del self.etc_model
        torch.cuda.empty_cache()
            
    def internal_predict_for_ti(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.ti_eval_batch_size + (len(eval_data) % self.config.ti_eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.ti_eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            batch_pred_spans = self.ti_model.predict(batch)
            for doc_id, wnd_id, tokens, text, pred_spans in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, 
                                                                   batch.batch_text, batch_pred_spans):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "triggers": pred_spans
                             }
                
                predictions.append(prediction)
        progress.close()
        
        return predictions
    
    def compute_etr_score(self, predictions, eval_data):
        hit_scores = []
        for prediction, data in zip(predictions, eval_data):
            gold_set = set([t[2] for t in data["triggers"]])
            preds = [t[2] for t in prediction["triggers"]]
            hit = sum([p in gold_set for p in preds])/min(len(gold_set), len(preds))
            hit_scores.append(hit*100.0)
        
        avg_score = np.mean(hit_scores)
        return avg_score
    
    def internal_predict_for_etr(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.etr_eval_batch_size + (len(eval_data) % self.config.etr_eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.etr_eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            batch_preds = self.etr_model.predict(batch)
            for doc_id, wnd_id, tokens, text, preds in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, 
                                                                   batch.batch_text, batch_preds):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "triggers": [(0, 0, t) for t in preds]
                             }
                predictions.append(prediction)
        progress.close()
        
        return predictions
    
    def compute_etc_score(self, predictions, eval_data):
        hit_scores = [prediction["etc_label"]==data["etc_label"] for prediction, data in zip(predictions, eval_data)]
        return np.mean(hit_scores)*100.0
    
    def internal_predict_for_etc(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.etr_eval_batch_size + (len(eval_data) % self.config.etr_eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.etr_eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            batch_preds = self.etc_model.predict(batch)
            for doc_id, wnd_id, tokens, text, triggers, spans, pred in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, 
                                                                           batch.batch_text, batch.batch_triggers, batch.batch_spans, batch_preds):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "triggers": triggers,
                              "spans": spans,
                              "etc_label": pred,
                             }
                predictions.append(prediction)
        progress.close()
        
        return predictions
    
    def predict(self, data, **kwargs):
        assert self.tokenizer and self.ti_model and self.etr_model and self.etc_model
        internal_data = self.process_data_for_ti(data)
        ti_predictions = self.internal_predict_for_ti(internal_data, split="Test TI")
        etr_predictions = self.internal_predict_for_etr(internal_data, split="Test ETR")
        
        sep = self.tokenizer.sep_token
        new_data = []
        for dt, ti_prediction, etr_prediction in zip(internal_data, ti_predictions, etr_predictions):
            for ti_pred in ti_prediction["triggers"]:
                for etr_pred in etr_prediction["triggers"]:
                    event_type = etr_pred[2]
                    definition = event_type_tags[self.config.dataset][event_type]
                    sentence = " ".join(dt["tokens"])
                    trigger_span = " ".join(dt["tokens"][ti_pred[0]:ti_pred[1]])
                    text = TEMPLATE.format(definition=definition, sentence=sentence, trigger=trigger_span, sep=sep)
                    new_dt = {"doc_id": dt["doc_id"], 
                              "wnd_id": dt["wnd_id"], 
                              "tokens": dt["tokens"], 
                              "pieces": dt["pieces"], 
                              "token_lens": dt["token_lens"], 
                              "token_num": dt["token_num"], 
                              "text": dt["text"], 
                              "triggers": [etr_pred],
                              "spans": [ti_pred],
                              "etc_text": text,
                              "etc_label": 0,
                             }
                    new_data.append(new_dt)
        
        etc_predictions = self.internal_predict_for_etc(new_data, split="Test ETC")
        
        tmp_preds = defaultdict(list)
        for etc_prediction in etc_predictions:
            if etc_prediction["etc_label"] == 1:
                trigger_pred = (etc_prediction["spans"][0][0], etc_prediction["spans"][0][1], etc_prediction["triggers"][0][2])
                tmp_preds[(etc_prediction["doc_id"], etc_prediction["wnd_id"])].append(trigger_pred)
                
        predictions = [] 
        for dt in internal_data:
            prediction = {"doc_id": dt["doc_id"],  
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "text": dt["text"], 
                          "triggers": tmp_preds[(dt["doc_id"], dt["wnd_id"])]
                         }

            predictions.append(prediction)
                
        return predictions
