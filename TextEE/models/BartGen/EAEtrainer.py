import os, sys, logging, tqdm, pprint, re
import torch
import numpy as np
from collections import namedtuple
from transformers import BartTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from collections import defaultdict
from ..trainer import BasicTrainer
from .EAEmodel import BartGenEAEModel
from .pattern import patterns
from scorer import compute_EAE_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

EAEBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_text', 'batch_piece_idxs', 'batch_token_start_idxs', 
                   'batch_trigger', 'batch_arguments', 'batch_input', 'batch_target']
EAEBatch = namedtuple('EAEBatch', field_names=EAEBatch_fields, defaults=[None] * len(EAEBatch_fields))

TRI_STR = '<tgr>'
ARG_STR = '<arg>'
AND_STR = 'and'

def EAE_collate_fn(batch):
    return EAEBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance["tokens"] for instance in batch], 
        batch_text=[instance["text"] for instance in batch], 
        batch_piece_idxs=[instance["piece_idxs"] for instance in batch], 
        batch_token_start_idxs=[instance["token_start_idxs"] for instance in batch], 
        batch_trigger=[instance["trigger"] for instance in batch], 
        batch_arguments=[instance["arguments"] for instance in batch], 
        batch_input=[instance["input"] for instance in batch], 
        batch_target=[instance["target"] for instance in batch], 
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
        
def generate_target_string(tokens, trigger, arguments, pattern_map):
    role_candidates = pattern_map[trigger[2]]["roles"]
    prompt_template = pattern_map[trigger[2]]["template"]
    for i, r in enumerate(role_candidates):
        prompt_template = prompt_template.replace(f"<arg{i+1}>", "<arg>")
    target = pattern_map[trigger[2]]["template"]
    
    arg_list = defaultdict(list)
    for r in sorted(arguments, key=lambda x: (x[0], x[1])):
        arg_list[r[2]].append(" ".join(tokens[r[0]:r[1]]))
    for r in role_candidates:
        if len(arg_list[r]) == 0:
            arg_list[r].append(ARG_STR)
    for i, r in enumerate(role_candidates):
        target = target.replace(f"<arg{i+1}>", f" {AND_STR} ".join(arg_list[r]))
        
    return target, prompt_template

def get_arg_objects(event_type, text, pattern_map):
    
    template_words = pattern_map[event_type]["template"].strip().split() + [""]
    predicted_words = text.strip().split()
    
    outputs = []
    t_ptr= 0
    p_ptr= 0 

    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = int(m.group(1)[3:]) - 1
            arg_name = pattern_map[event_type]["roles"][arg_num]

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr +=1 
                t_ptr +=1  
            else:
                arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) and (predicted_words[p_ptr] != template_words[t_ptr+1]):
                    p_ptr+=1 
                arg_text = " ".join(predicted_words[arg_start:p_ptr])
                for arg_t in arg_text.split(f" {AND_STR} "):
                    outputs.append((arg_t, arg_name)) 
                t_ptr+=1 
                # aligned 
        else:
            t_ptr+=1 
            p_ptr+=1 
            
    return outputs

class BartGenEAETrainer(BasicTrainer):
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
            self.model = BartGenEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('facebook/bart-'):
                self.tokenizer = BartTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, use_fast=False)
            
            special_tokens = [f'{ARG_STR}', f'{TRI_STR}']
            logger.info(f"Add tokens {special_tokens}")
            self.tokenizer.add_tokens(special_tokens)
            self.model = BartGenEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)
            
        self.generate_vocab()
            
    def generate_vocab(self):
        event_type_itos = sorted(self.type_set["trigger"])
        event_type_stoi = {e: i for i, e in enumerate(event_type_itos)}
        role_type_itos = sorted(self.type_set["role"])
        role_type_stoi = {r: i for i, r in enumerate(role_type_itos)}
        self.vocab = {"event_type_itos": event_type_itos, 
                      "event_type_stoi": event_type_stoi,
                      "role_type_itos": role_type_itos,
                      "role_type_stoi": role_type_stoi,
                     }
    
    def process_data(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        n_total = 0
        new_data = []
        for dt in data:
            
            n_total += 1
            
            target, prompt_template = generate_target_string(dt["tokens"], dt["trigger"], dt["arguments"], patterns[self.config.dataset])
            
            tri_start, tri_end = dt["trigger"][0], dt["trigger"][1]
            input_with_tri = " ".join(dt["tokens"][:tri_start] + [TRI_STR] + dt["tokens"][tri_start:tri_end] + [TRI_STR] + dt["tokens"][tri_end:])
            
            input_str = f"{self.tokenizer.bos_token} {prompt_template} {self.tokenizer.bos_token} {self.tokenizer.eos_token} {input_with_tri} {self.tokenizer.eos_token}"
            if len(self.tokenizer.tokenize(input_str)) > self.config.max_length:
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
                      "trigger": dt["trigger"], 
                      "arguments": dt["arguments"], 
                      "input": input_str, 
                      "target": target, 
                     }
            new_data.append(new_dt)
                
        logger.info(f"Generate {len(new_data)} BartGen EAE instances from {n_total} EAE instances")

        return new_data

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data(train_data)
        internal_dev_data = self.process_data(dev_data)
        
        param_groups = [{'params': self.model.parameters(), 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay}]
        
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        optimizer = AdamW(params=param_groups)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.max_epoch)
        
        best_scores = {"argument_attached_cls": {"f1": 0.0}}
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
                                                         shuffle=True, drop_last=False, collate_fn=EAE_collate_fn)):
                
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
            dev_scores = compute_EAE_scores(predictions, internal_dev_data, metrics={"argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"})

            # print scores
            print(f"Dev {epoch}")
            print_scores(dev_scores)
            
            if dev_scores["argument_attached_cls"]["f1"] >= best_scores["argument_attached_cls"]["f1"]:
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
                                                     shuffle=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            pred_texts = self.model.predict(batch, num_beams=self.config.beam_size, max_length=self.config.max_output_length)
            for doc_id, wnd_id, tokens, text, piece_idxs, token_start_idxs, trigger, pred_text in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, batch.batch_text, 
                                                                                                      batch.batch_piece_idxs, batch.batch_token_start_idxs, 
                                                                                                      batch.batch_trigger, pred_texts):
                
                pred_objects = get_arg_objects(trigger[2], pred_text, patterns[self.config.dataset])
                
                pred_arguments = []
                for span, role_type in pred_objects:
                    sid, eid = get_span_idx(piece_idxs, token_start_idxs, span, self.tokenizer, trigger_span=trigger[:2])
                    if sid == -1:
                        continue
                    pred_arguments.append((sid, eid, role_type, span))
                
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
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
