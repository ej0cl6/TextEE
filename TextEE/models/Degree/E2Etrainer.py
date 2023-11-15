import os, sys, logging, tqdm, pprint
import torch
import numpy as np
from collections import namedtuple
from transformers import BartTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .E2Emodel import DegreeE2EModel
from .template_generate import event_template, eve_template_generator
from .pattern import patterns, ROLE_PH_MAP
from scorer import compute_f1, print_scores
import ipdb

logger = logging.getLogger(__name__)

EDBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_text', 'batch_piece_idxs', 'batch_token_start_idxs', 
                   'batch_input', 'batch_target', 'batch_info']
EDBatch = namedtuple('EDBatch', field_names=EDBatch_fields, defaults=[None] * len(EDBatch_fields))

def ED_collate_fn(batch):
    return EDBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance["tokens"] for instance in batch], 
        batch_text=[instance["text"] for instance in batch], 
        batch_piece_idxs=[instance["piece_idxs"] for instance in batch], 
        batch_token_start_idxs=[instance["token_start_idxs"] for instance in batch], 
        batch_input=[instance["input"] for instance in batch], 
        batch_target=[instance["target"] for instance in batch], 
        batch_info=[instance["info"] for instance in batch], 
    )

def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.
    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 
    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]

def get_span_idx_tri(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.
    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 
    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return [(-1, -1)]
    else:
        if trigger_span is None:
            return candidates
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))

class DegreeE2ETrainer(BasicTrainer):
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
            self.model = DegreeE2EModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('facebook/bart-'):
                self.tokenizer = BartTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, use_fast=False)
            
            special_tokens = ['<Trigger>', '<sep>', '<None>']
            logger.info(f"Add tokens {special_tokens}")
            self.tokenizer.add_tokens(special_tokens)
            self.model = DegreeE2EModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)
            
        self.generate_vocab()
            
    def generate_vocab(self):
        event_type_itos = sorted(self.type_set["trigger"])
        event_type_stoi = {e: i for i, e in enumerate(event_type_itos)}
        self.vocab = {"event_type_itos": event_type_itos, 
                      "event_type_stoi": event_type_stoi,
                     }
    
    def process_data_for_training(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        logger.info("Processing data...")
        
        n_total = 0
        new_data = []
        for dt in tqdm.tqdm(data, ncols=100):
            
            n_total += 1
            
            _triggers = [t["trigger"][:3] for t in dt["events"]]
            _arguments = [(t["trigger"][:3], r[:3]) for t in dt["events"] for r in t["arguments"]]
            event_template = eve_template_generator(self.config.dataset, dt["tokens"], _triggers, _arguments, self.config.input_style, self.config.output_style, self.vocab, True)
            
            all_data_ = event_template.get_training_data()
            pos_data_ = [data_ for data_ in all_data_ if data_[3]]
            neg_data_ = [data_ for data_ in all_data_ if not data_[3]]
            np.random.shuffle(neg_data_)
            
            for data_ in pos_data_:
                if len(self.tokenizer.tokenize(data_[0])) > self.config.max_length:
                    continue
                    
                pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
                token_lens = [len(p) for p in pieces]
                pieces = [p for piece in pieces for p in piece]
                piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
                assert sum(token_lens) == len(piece_idxs)
                token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]

                new_dt = {"doc_id": dt["doc_id"], 
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "text": dt["text"], 
                          "piece_idxs": piece_idxs, 
                          "token_start_idxs": token_start_idxs,
                          "input": data_[0], 
                          "target": data_[1], 
                          "info": (data_[2], data_[4], data_[5])
                         }
                new_data.append(new_dt)

            for data_ in neg_data_[:self.config.n_negative]:
                if len(self.tokenizer.tokenize(data_[0])) > self.config.max_length:
                    continue
                    
                pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
                token_lens = [len(p) for p in pieces]
                pieces = [p for piece in pieces for p in piece]
                piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
                assert sum(token_lens) == len(piece_idxs)
                token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]

                new_dt = {"doc_id": dt["doc_id"], 
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "text": dt["text"], 
                          "piece_idxs": piece_idxs, 
                          "token_start_idxs": token_start_idxs,
                          "input": data_[0], 
                          "target": data_[1], 
                          "info": (data_[2], data_[4], data_[5])
                         }
                new_data.append(new_dt)
            
        logger.info(f"Generate {len(new_data)} Degree E2E instances from {n_total} E2E instances")

        return new_data
    
    def process_data_for_testing(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        n_total = 0
        new_data = []
        for dt in data:
            
            n_total += 1
            
            pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            pieces = [p for piece in pieces for p in piece]
            piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
            assert sum(token_lens) == len(piece_idxs)
            token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]

            new_dt = {"doc_id": dt["doc_id"], 
                      "wnd_id": dt["wnd_id"], 
                      "tokens": dt["tokens"], 
                      "text": dt["text"], 
                      "piece_idxs": piece_idxs, 
                      "token_start_idxs": token_start_idxs,
                      "input": "", 
                      "target": "", 
                      "info": "",
                     }
            new_data.append(new_dt)

        logger.info(f"Generate {len(new_data)} Degree E2E instances from {n_total} E2E instances")

        return new_data

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data_for_training(train_data)
        internal_dev_data = self.process_data_for_training(dev_data)
        
        param_groups = [{'params': self.model.parameters(), 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay}]
        
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        optimizer = AdamW(params=param_groups)
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
            dev_scores = self.internal_eval(internal_dev_data, split="Dev")

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
        
        
    def internal_eval(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        eval_gold_tri_num, eval_pred_tri_num, eval_match_tri_num = 0, 0, 0
        eval_gold_arg_num, eval_pred_arg_num, eval_match_arg_id, eval_match_arg_cls = 0, 0, 0, 0
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            pred_texts = self.model.predict(batch, num_beams=self.config.beam_size, max_length=self.config.max_output_length)
            for doc_id, wnd_id, tokens, text, piece_idxs, token_start_idxs, info, pred_text in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, batch.batch_text, 
                                                                                                      batch.batch_piece_idxs, batch.batch_token_start_idxs, 
                                                                                                      batch.batch_info, pred_texts):
                
                template = event_template(info[1], patterns[self.config.dataset][info[1]], self.config.input_style, self.config.output_style, tokens, ROLE_PH_MAP[self.config.dataset], info[0])
                pred_objects = template.decode(pred_text)
                
                # calculate scores
                sub_scores = template.evaluate(pred_objects)
                eval_gold_tri_num += sub_scores['gold_tri_num']
                eval_pred_tri_num += sub_scores['pred_tri_num']
                eval_match_tri_num += sub_scores['match_tri_num']
                eval_gold_arg_num += sub_scores['gold_arg_num']
                eval_pred_arg_num += sub_scores['pred_arg_num']
                eval_match_arg_id += sub_scores['match_arg_id']
                eval_match_arg_cls += sub_scores['match_arg_cls']
                
        progress.close()
        
        tri_id_scores = compute_f1(eval_pred_tri_num, eval_gold_tri_num, eval_match_tri_num)
        arg_id_scores = compute_f1(eval_pred_arg_num, eval_gold_arg_num, eval_match_arg_id)
        arg_cls_scores = compute_f1(eval_pred_arg_num, eval_gold_arg_num, eval_match_arg_cls)
        
        eval_scores = {
            "trigger_id": {
                "pred_num": eval_pred_tri_num, 
                "gold_num": eval_gold_tri_num, 
                "match_num": eval_match_tri_num, 
                "precision": tri_id_scores[0], 
                "recall": tri_id_scores[1], 
                "f1": tri_id_scores[2], 
            },
            "argument_id": {
                "pred_num": eval_pred_arg_num, 
                "gold_num": eval_gold_arg_num, 
                "match_num": eval_match_arg_id, 
                "precision": arg_id_scores[0], 
                "recall": arg_id_scores[1], 
                "f1": arg_id_scores[2], 
            },
            "argument_cls": {
                "pred_num": eval_pred_arg_num, 
                "gold_num": eval_gold_arg_num, 
                "match_num": eval_match_arg_cls, 
                "precision": arg_cls_scores[0], 
                "recall": arg_cls_scores[1], 
                "f1": arg_cls_scores[2], 
            }
        }
        
        return eval_scores
    
    def predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        
        eval_data = self.process_data_for_testing(data)
        
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc="Test")
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            
            batch_pred_events = [[] for _ in range(len(batch.batch_tokens))]
            for event_type in self.vocab['event_type_itos']:
                inputs = []
                for tokens in batch.batch_tokens:
                    template = event_template(event_type, patterns[self.config.dataset][event_type], self.config.input_style, self.config.output_style, tokens, ROLE_PH_MAP[self.config.dataset])
                    inputs.append(template.generate_input_str(''))
                    
                
                inputs = self.tokenizer(inputs, return_tensors='pt', padding=True)
                enc_idxs = inputs['input_ids'].cuda()
                enc_attn = inputs['attention_mask'].cuda()

                pred_texts = self.model.generate(enc_idxs, enc_attn, num_beams=self.config.beam_size, max_length=self.config.max_output_length)
                
                for bid, (tokens, piece_idxs, token_start_idxs, pred_text) in enumerate(zip(batch.batch_tokens, batch.batch_piece_idxs, batch.batch_token_start_idxs, pred_texts)):
                    template = event_template(event_type, patterns[self.config.dataset][event_type], self.config.input_style, self.config.output_style, tokens, ROLE_PH_MAP[self.config.dataset])
                    pred_objects = template.decode(pred_text)
                    
                    pred_trigger_objects = [obj for obj in pred_objects if obj[1] == event_type]
                    pred_argument_objects = [obj for obj in pred_objects if obj[1] != event_type]
                    
                    # decode triggers
                    triggers_ = [r+(event_type, kwargs) for span, _, kwargs in pred_trigger_objects for r in get_span_idx_tri(piece_idxs, token_start_idxs, span, self.tokenizer)]
                    triggers_ = [t for t in triggers_ if t[0] != -1]
                    # triggers_ = list(set(triggers_))
                    
                    # decode arguments
                    tri_id2obj = {}
                    arguments_ = {}
                    for t in triggers_:
                        try:
                            tri_id2obj[t[3]['tri counter']] = (t[0], t[1], t[2])
                            arguments_[(t[0], t[1], t[2])] = []
                        except:
                            ipdb.set_trace()

                    for span, role_type, kwargs in pred_argument_objects:
                        corres_tri_id = kwargs['cor tri cnt']
                        if corres_tri_id in tri_id2obj.keys():
                            arg_span = get_span_idx(piece_idxs, token_start_idxs, span, self.tokenizer, tri_id2obj[corres_tri_id])
                            if arg_span[0] != -1:
                                arguments_[tri_id2obj[corres_tri_id]].append((arg_span[0], arg_span[1], role_type))
                        else:
                            arg_span = get_span_idx(piece_idxs, token_start_idxs, span, self.tokenizer)
                            if arg_span[0] != -1:
                                if len(triggers_) > 0:
                                    arguments_[triggers_[0][:3]].append((arg_span[0], arg_span[1], role_type))
                                    
                    for trigger_ in triggers_:
                        event = {"trigger": trigger_[:3], "arguments": arguments_[trigger_[:3]]}
                        batch_pred_events[bid].append(event)
                    
            for doc_id, wnd_id, tokens, text, pred_events in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, 
                                                                   batch.batch_text, batch_pred_events):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "events": pred_events
                             }
                
                predictions.append(prediction)
                
        progress.close()
        
        return predictions
    
    def predictEAE(self, data, **kwargs):
        raise ValueError(f"Degree-E2E does not support EAE only predictions.")
        
    def predictEARL(self, data, **kwargs):
        raise ValueError(f"Degree-E2E does not support EARL only predictions.")
        