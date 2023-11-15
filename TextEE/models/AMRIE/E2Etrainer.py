import os, sys, logging, tqdm, pprint, copy
import torch
import numpy as np
from transformers import (BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer,
                          AutoTokenizer, AdamW, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .E2Emodel import AMRIEE2EModel
from .data import IEDataset
from .util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task
from .scorer import score_graphs
from scorer import compute_f1, print_scores
import ipdb

logger = logging.getLogger(__name__)

class AMRIEE2ETrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.model = None
        self.valid_patterns = None
        
    @classmethod
    def add_extra_info_fn(cls, instances, raw_data, config):
        extra_info_map = {}
        for dt in raw_data:
            extra_info = {
                "entity_mentions": dt["entity_mentions"] if "entity_mentions" in dt else [], 
                "relation_mentions": dt["relation_mentions"] if "relation_mentions" in dt else [], 
                "event_mentions": dt["event_mentions"] if "event_mentions" in dt else [], 
            }
            extra_info_map[(dt["doc_id"], dt["wnd_id"])] = extra_info
        for instance in instances:
            instance["extra_info"] = extra_info_map[(instance["doc_id"], instance["wnd_id"])]
        
        return instances
    
    def get_idx_map(self, tokens1, tokens2):
        len1, len2 = len(tokens1), len(tokens2)
        idx1_s, idx1_e, idx2_s, idx2_e, = 0, 0, 0, 0
        idx_map = np.zeros((len2+1, ), dtype=np.int32)
        idx_map[-1] = len1
        while idx1_e <= len1 and idx2_e <= len2:
            if "".join(tokens1[idx1_s:idx1_e+1]) == "".join(tokens2[idx2_s:idx2_e+1]):
                idx_map[idx2_s:idx2_e+1] = idx1_s
                idx1_s = idx1_e+1
                idx1_e = idx1_e+1
                idx2_s = idx2_e+1
                idx2_e = idx2_e+1
            elif len("".join(tokens1[idx1_s:idx1_e+1])) <= len("".join(tokens2[idx2_s:idx2_e+1])):
                idx1_e += 1
            else:
                idx2_e += 1
        return idx_map
        
    def load_tokenizer_(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading tokenizer from {checkpoint}")
            state = torch.load(os.path.join(checkpoint, "best_model.tokenizer"))
            self.tokenizer = state["tokenizer"]
        else:
            logger.info(f"Loading tokenizer from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('bert-'):
                self.tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            elif self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            elif self.config.pretrained_model_name.startswith('xlm-roberta-'):
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, do_lower_case=False)
                
    def load_model_(self, checkpoint=None):
        assert self.tokenizer
        
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(os.path.join(checkpoint, "best_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            self.vocabs = state["vocabs"]
            self.type_set = state["type_set"]
            self.valid_patterns = state["valid_patterns"]
            self.model = AMRIEE2EModel(self.config, self.vocabs, self.valid_patterns)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            self.valid_patterns = load_valid_patterns(self.config.valid_pattern_path, self.vocabs)
            self.model = AMRIEE2EModel(self.config, self.vocabs, self.valid_patterns)
            self.model.cuda(device=self.config.gpu_device)
    
    def load_model(self, checkpoint=None):
        self.load_tokenizer_(checkpoint=checkpoint)
        self.load_model_(checkpoint=checkpoint)
            
    def train(self, train_data, dev_data, **kwargs):
        
        logger.info("Loading graphs")
        org_train_graphs, train_align, train_exist = torch.load(self.config.processed_train_amr)
        org_dev_graphs, dev_align, dev_exist = torch.load(self.config.processed_dev_amr)
        
        train_graphs, dev_graphs = [], []
        for g in org_train_graphs:
            g_device = g.to(self.config.gpu_device)
            train_graphs.append(g_device)
        for g in org_dev_graphs:
            g_device = g.to(self.config.gpu_device)
            dev_graphs.append(g_device)
        
        self.load_tokenizer_()
        
        train_set = IEDataset(train_data, self.tokenizer, train_graphs, train_align, train_exist, gpu=True,
                              max_length=self.config.max_length,
                              relation_mask_self=self.config.relation_mask_self,
                              relation_directional=self.config.relation_directional,
                              symmetric_relations=self.config.symmetric_relations)
        
        dev_set = IEDataset(dev_data, self.tokenizer, dev_graphs, dev_align, dev_exist, gpu=True,
                              max_length=self.config.max_length,
                              relation_mask_self=self.config.relation_mask_self,
                              relation_directional=self.config.relation_directional,
                              symmetric_relations=self.config.symmetric_relations)
        
        self.vocabs = generate_vocabs([train_set, dev_set])
        
        train_set.numberize(self.tokenizer, self.vocabs)
        dev_set.numberize(self.tokenizer, self.vocabs)
        
        self.load_model_()
        
        batch_num = len(train_set) // self.config.batch_size + (len(train_set) % self.config.batch_size != 0)
        dev_batch_num = len(dev_set) // self.config.eval_batch_size + (len(dev_set) % self.config.eval_batch_size != 0)
        
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if n.startswith('bert')],
                'lr': self.config.bert_learning_rate, 'weight_decay': self.config.bert_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('bert')
                           and 'crf' not in n and 'global_feature' not in n],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('bert')
                           and ('crf' in n or 'global_feature' in n)],
                'lr': self.config.learning_rate, 'weight_decay': 0
            }
        ]
        optimizer = AdamW(params=param_groups)
        schedule = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=batch_num*self.config.warmup_epoch,
                                                   num_training_steps=batch_num*self.config.max_epoch)
        
        best_scores = {self.config.target_task: {"f": 0.0}}
        best_epoch = -1
        target_task = self.config.target_task
        
        logger.info('================Start Training================')
        for epoch in range(self.config.max_epoch):
            logger.info('Epoch: {}'.format(epoch)) 
            # training step
            progress = tqdm.tqdm(total=batch_num, ncols=75,
                                 desc='Train {}'.format(epoch))
            optimizer.zero_grad()
            cummulate_loss = 0.
            for batch_idx, batch in enumerate(DataLoader(
                    train_set, batch_size=self.config.batch_size // self.config.accumulate_step,
                    shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
                loss = self.model(batch, epoch)
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss += loss
                loss.backward()
                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clipping)
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad()
                    
            progress.close()
            logger.info({"average training loss": (cummulate_loss / batch_idx).data})

            # dev set
            progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                                 desc='Dev {}'.format(epoch))
            best_dev_role_model = False
            dev_gold_graphs, dev_pred_graphs, dev_tokens = [], [], []
            for batch in DataLoader(dev_set, batch_size=self.config.eval_batch_size,
                                    shuffle=False, collate_fn=dev_set.collate_fn):
                progress.update(1)
                graphs = self.model.predict(batch, epoch, gold_tri=False)
                gold_graph = copy.deepcopy(batch.graphs)
                for graph in graphs:
                    graph.clean(relation_directional=self.config.relation_directional,
                                symmetric_relations=self.config.symmetric_relations)
                dev_gold_graphs.extend(gold_graph)
                dev_pred_graphs.extend(graphs)
                dev_tokens.extend(batch.tokens)
                
            progress.close()
            
            dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs, self.vocabs['event_type'])
            
            if dev_scores[target_task]['f'] >= best_scores[target_task]['f']:
                best_scores = dev_scores
                logger.info('Saving best model')
                state = dict(model=self.model.state_dict(), vocabs=self.vocabs, type_set=self.type_set, valid_patterns=self.valid_patterns)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                state = dict(tokenizer=self.tokenizer)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.tokenizer"))
                best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))
    
    def predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        
        logger.info("Loading graphs")
        org_eval_graphs, eval_align, eval_exist = torch.load(kwargs["processed_test_amr"])
        eval_graphs = []
        for g in org_eval_graphs:
            g_device = g.to(self.config.gpu_device)
            eval_graphs.append(g_device)
        
        eval_set = IEDataset(data, self.tokenizer, eval_graphs, eval_align, eval_exist, gpu=True,
                             max_length=self.config.max_length,
                             relation_mask_self=self.config.relation_mask_self,
                             relation_directional=self.config.relation_directional,
                             symmetric_relations=self.config.symmetric_relations, 
                             test=True
                            )
        
        eval_set.numberize(self.tokenizer, self.vocabs)
        
        eval_batch_num = len(eval_set) // self.config.eval_batch_size + (len(eval_set) % self.config.eval_batch_size != 0)
        
        progress = tqdm.tqdm(total=eval_batch_num, ncols=75,
                                 desc='Test')

        eval_gold_graphs, eval_pred_graphs, eval_tokens = [], [], []
        for batch in DataLoader(eval_set, batch_size=self.config.eval_batch_size,
                                shuffle=False, collate_fn=eval_set.collate_fn):
            progress.update(1)
            graphs = self.model.predict(batch, 0, gold_tri=False)
            gold_graph = copy.deepcopy(batch.graphs)
            for graph in graphs:
                    graph.clean(relation_directional=self.config.relation_directional,
                                symmetric_relations=self.config.symmetric_relations)
            eval_gold_graphs.extend(gold_graph)
            eval_pred_graphs.extend(graphs)
            eval_tokens.extend(batch.tokens)
        progress.close()
        
        event_type_itos = {i: e for e, i in self.vocabs["event_type"].items()}
        role_type_itos = {i: r for r, i in self.vocabs["role_type"].items()}
        
        assert len(data) == len(eval_pred_graphs)
        
        predictions = []
        for dt, dti, pred_graph in zip(data, eval_set, eval_pred_graphs):
        
            idx_map = self.get_idx_map(dt["tokens"], dti.tokens)
            
            pred_entities = [(idx_map[e[0]], idx_map[e[1]]) for e in pred_graph.entities]
            pred_triggers = [(idx_map[t[0]], idx_map[t[1]], event_type_itos[t[2]]) for t in pred_graph.triggers]
            pred_arguments = [[] for t in pred_triggers]
            for argument in pred_graph.roles:
                map_e = pred_entities[argument[1]]
                role_type = role_type_itos[argument[2]]
                pred_arguments[argument[0]].append((map_e[0], map_e[1], role_type))
                
            pred_events = [{"trigger": t, "arguments": a} for t, a in zip(pred_triggers, pred_arguments)]
            if (dt["doc_id"], dt["wnd_id"]) in eval_set.skip_insts:
                pred_events = []
                
            prediction = {"doc_id": dt["doc_id"],  
                          "wnd_id": dt["wnd_id"],  
                          "tokens": dt["tokens"], 
                          "events": pred_events
                         }
            predictions.append(prediction)
        
        return predictions
    
    def predictEAE(self, data, **kwargs):
        assert self.tokenizer and self.model
        
        logger.info("Loading graphs")
        org_eval_graphs, eval_align, eval_exist = torch.load(kwargs["processed_test_amr"])
        eval_graphs = []
        for g in org_eval_graphs:
            g_device = g.to(self.config.gpu_device)
            eval_graphs.append(g_device)
        
        eval_set = IEDataset(data, self.tokenizer, eval_graphs, eval_align, eval_exist, gpu=True,
                             max_length=self.config.max_length,
                             relation_mask_self=self.config.relation_mask_self,
                             relation_directional=self.config.relation_directional,
                             symmetric_relations=self.config.symmetric_relations, 
                             test=True
                            )
        
        eval_set.numberize(self.tokenizer, self.vocabs)
        
        eval_batch_num = len(eval_set) // self.config.eval_batch_size + (len(eval_set) % self.config.eval_batch_size != 0)
        
        progress = tqdm.tqdm(total=eval_batch_num, ncols=75,
                                 desc='Test')

        eval_gold_graphs, eval_pred_graphs, eval_tokens = [], [], []
        for batch in DataLoader(eval_set, batch_size=self.config.eval_batch_size,
                                shuffle=False, collate_fn=eval_set.collate_fn):
            progress.update(1)
            graphs = self.model.predict(batch, 0, gold_tri=True)
            gold_graph = copy.deepcopy(batch.graphs)
            for graph in graphs:
                    graph.clean(relation_directional=self.config.relation_directional,
                                symmetric_relations=self.config.symmetric_relations)
            eval_gold_graphs.extend(gold_graph)
            eval_pred_graphs.extend(graphs)
            eval_tokens.extend(batch.tokens)
        progress.close()
        
        event_type_itos = {i: e for e, i in self.vocabs["event_type"].items()}
        role_type_itos = {i: r for r, i in self.vocabs["role_type"].items()}
        
        assert len(data) == len(eval_pred_graphs)
        
        predictions = []
        for dt, dti, pred_graph in zip(data, eval_set, eval_pred_graphs):
        
            idx_map = self.get_idx_map(dt["tokens"], dti.tokens)
            
            pred_entities = [(idx_map[e[0]], idx_map[e[1]]) for e in pred_graph.entities]
            pred_triggers = [(idx_map[t[0]], idx_map[t[1]], event_type_itos[t[2]]) for t in pred_graph.triggers]
            pred_arguments = [[] for t in pred_triggers]
            for argument in pred_graph.roles:
                map_e = pred_entities[argument[1]]
                role_type = role_type_itos[argument[2]]
                pred_arguments[argument[0]].append((map_e[0], map_e[1], role_type))
                
            pred_events = [{"trigger": t, "arguments": a} for t, a in zip(pred_triggers, pred_arguments)]
            if (dt["doc_id"], dt["wnd_id"]) in eval_set.skip_insts:
                pred_events = []
            prediction = {"doc_id": dt["doc_id"],  
                          "wnd_id": dt["wnd_id"],  
                          "tokens": dt["tokens"], 
                          "events": pred_events
                         }
            predictions.append(prediction)
        
        return predictions
    
    def predictEARL(self, data, **kwargs):
        assert self.tokenizer and self.model
        
        logger.info("Loading graphs")
        org_eval_graphs, eval_align, eval_exist = torch.load(kwargs["processed_test_amr"])
        eval_graphs = []
        for g in org_eval_graphs:
            g_device = g.to(self.config.gpu_device)
            eval_graphs.append(g_device)
        
        eval_set = IEDataset(data, self.tokenizer, eval_graphs, eval_align, eval_exist, gpu=True,
                             max_length=self.config.max_length,
                             relation_mask_self=self.config.relation_mask_self,
                             relation_directional=self.config.relation_directional,
                             symmetric_relations=self.config.symmetric_relations, 
                             test=True
                            )
        
        eval_set.numberize(self.tokenizer, self.vocabs)
        
        eval_batch_num = len(eval_set) // self.config.eval_batch_size + (len(eval_set) % self.config.eval_batch_size != 0)
        
        progress = tqdm.tqdm(total=eval_batch_num, ncols=75,
                                 desc='Test')

        eval_gold_graphs, eval_pred_graphs, eval_tokens = [], [], []
        for batch in DataLoader(eval_set, batch_size=self.config.eval_batch_size,
                                shuffle=False, collate_fn=eval_set.collate_fn):
            progress.update(1)
            graphs = self.model.predict(batch, 0, gold_tri=True, gold_ent=True)
            gold_graph = copy.deepcopy(batch.graphs)
            for graph in graphs:
                    graph.clean(relation_directional=self.config.relation_directional,
                                symmetric_relations=self.config.symmetric_relations)
            eval_gold_graphs.extend(gold_graph)
            eval_pred_graphs.extend(graphs)
            eval_tokens.extend(batch.tokens)
        progress.close()
        
        event_type_itos = {i: e for e, i in self.vocabs["event_type"].items()}
        role_type_itos = {i: r for r, i in self.vocabs["role_type"].items()}
        
        assert len(data) == len(eval_pred_graphs)
        
        predictions = []
        for dt, dti, pred_graph in zip(data, eval_set, eval_pred_graphs):
        
            idx_map = self.get_idx_map(dt["tokens"], dti.tokens)
            
            pred_entities = [(idx_map[e[0]], idx_map[e[1]]) for e in pred_graph.entities]
            pred_triggers = [(idx_map[t[0]], idx_map[t[1]], event_type_itos[t[2]]) for t in pred_graph.triggers]
            pred_arguments = [[] for t in pred_triggers]
            for argument in pred_graph.roles:
                map_e = pred_entities[argument[1]]
                role_type = role_type_itos[argument[2]]
                pred_arguments[argument[0]].append((map_e[0], map_e[1], role_type))
                
            pred_events = [{"trigger": t, "arguments": a} for t, a in zip(pred_triggers, pred_arguments)]
            if (dt["doc_id"], dt["wnd_id"]) in eval_set.skip_insts:
                pred_events = []
            prediction = {"doc_id": dt["doc_id"],  
                          "wnd_id": dt["wnd_id"],  
                          "tokens": dt["tokens"], 
                          "events": pred_events
                         }
            predictions.append(prediction)
        
        return predictions
