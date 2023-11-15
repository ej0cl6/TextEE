import csv, os, re, json, jsonlines
import torch
from .utils import EXTERNAL_TOKENS
import logging
import ipdb

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id, 
                event_type, event_trigger,
                enc_text, enc_input_ids, enc_mask_ids, 
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                arg_quries, arg_joint_prompt, target_info,
                old_tok_to_new_tok_index = None, full_text = None, arg_list=None

        ):

        self.example_id = example_id
        self.feature_id = feature_id
        self.event_type = event_type
        self.event_trigger = event_trigger
        
        self.enc_text = enc_text
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids

        self.dec_prompt_texts = dec_prompt_text
        self.dec_prompt_ids =dec_prompt_ids
        self.dec_prompt_mask_ids=dec_prompt_mask_ids

        assert arg_quries == None
        
        self.arg_joint_prompt = arg_joint_prompt
        
        self.target_info = target_info
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index

        self.full_text = full_text
        self.arg_list = arg_list


    def init_pred(self):
        self.pred_dict_tok = dict()
        self.pred_dict_word = dict()
    

    def add_pred(self, role, span, dset_type):
        if role not in self.pred_dict_tok:
            self.pred_dict_tok[role] = list()
        if span not in self.pred_dict_tok[role]:
            self.pred_dict_tok[role].append(span)

            if span!=(0, 0):
                if role not in self.pred_dict_word:
                    self.pred_dict_word[role] = list()
                word_span = self.get_word_span(span, dset_type)         # convert token span to word span 
                if word_span not in self.pred_dict_word[role]:
                    self.pred_dict_word[role].append(word_span)


    def set_gt(self, dset_type):
        self.gt_dict_tok = dict()
        for k,v in self.target_info.items():
            self.gt_dict_tok[k] = [(s,e) for (s,e) in zip(v["span_s"], v["span_e"])]

        self.gt_dict_word = dict()
        for role in self.gt_dict_tok:
            for span in self.gt_dict_tok[role]:
                if span!=(0, 0):
                    if role not in self.gt_dict_word:
                        self.gt_dict_word[role] = list()
                    word_span = self.get_word_span(span, dset_type)
                    self.gt_dict_word[role].append(word_span)

        
    @property
    def old_tok_index(self):
        new_tok_index_to_old_tok_index = dict()
        for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
            for j in range(new_tok_id_s, new_tok_id_e):
                new_tok_index_to_old_tok_index[j] = old_tok_id 
        return new_tok_index_to_old_tok_index


    def get_word_span(self, span, dset_type):
        """
        Given features with gt/pred token-spans, output gt/pred word-spans
        """
        if span==(0, 0):
            raise AssertionError()
        offset = 0 if dset_type=='ace05e' else self.event_trigger[2]
        span = list(span)
        span[0] = min(span[0], max(self.old_tok_index.keys()))
        span[1] = max(span[1]-1, min(self.old_tok_index.keys()))

        while span[0] not in self.old_tok_index:
            span[0] += 1 
        span_s = self.old_tok_index[span[0]] + offset
        while span[1] not in self.old_tok_index:
            span[1] -= 1 
        span_e = self.old_tok_index[span[1]] + offset
        while span_e < span_s:
            span_e += 1
        return (span_s, span_e + 1)

        
    def __repr__(self):
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
        s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
        return s


class DSET_processor:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.template_dict, self.argument_dict = self._read_roles(self.config.role_path)

    def _read_jsonlines(self, input_file):
        lines = []
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines


    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.load(f)


    def _read_roles(self, role_path):
        template_dict = {}
        role_dict = {}

        with open(role_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template
                
                event_type, arg = event_type_arg.split('_', 1)
                
                if event_type not in role_dict:
                    role_dict[event_type] = []
                role_dict[event_type].append(arg)
                
        return template_dict, role_dict

    def convert_EAEbatch_to_features(self, EAEbatch):
        pass

    def convert_features_to_batch(self, features):
        pass

    def generate_batch(self, EAEbatch):
        features = self.convert_EAEbatch_to_features(EAEbatch)
        batch = self.convert_features_to_batch(features)
        return features, batch

# change it to process only a batch (not generating dataloader!)
class MultiargProcessor(DSET_processor):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer) 

    @staticmethod
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

    def convert_EAEbatch_to_features(self, batch):
        prompts = self._read_prompt_group(self.config.prompt_path)

        if os.environ.get("DEBUG", False): counter = [0, 0, 0]
        features = []
        for doc_id, tokens, trigger, arguments in zip(batch.batch_doc_id, batch.batch_tokens, batch.batch_trigger, batch.batch_arguments):

            # NOTE: extend trigger full info in features
            trigger_start, trigger_end = trigger[0], trigger[1]
            event_trigger = [trigger[3], [trigger[0], trigger[1]], 0]
            event_type = trigger[2]

            event_args_name = [arg[2] for arg in arguments]
            if os.environ.get("DEBUG", False): counter[2] += len(event_args_name)
            sent = tokens[:trigger_start] + ['<t>'] + tokens[trigger_start:trigger_end] + ['</t>'] + tokens[trigger_end:]
            enc_text = " ".join(sent)

            # change the mapping to idx2tuple (start/end word idx)
            old_tok_to_char_index = []     # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART
            
            curr = 0
            for tok in sent:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append([curr, curr+len(tok)-1]) # exact word start char and end char index
                curr += len(tok)+1

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids)> self.config.max_enc_seq_length:
                raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")
            while len(enc_input_ids) < self.config.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.config.pad_mask_token)
            
            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
                if enc.char_to_token(char_idx_s) is None or enc.char_to_token(char_idx_e) is None:
                    new_tok_s = 0
                    new_tok_e = 1
                else:
                    new_tok_s = enc.char_to_token(char_idx_s)
                    new_tok_e = enc.char_to_token(char_idx_e) + 1
                    
                new_tok = [new_tok_s, new_tok_e]
                old_tok_to_new_tok_index.append(new_tok)    

            dec_prompt_text = prompts[event_type.replace(':', '.').replace('_', '.')].strip()
            if dec_prompt_text:
                dec_prompt = self.tokenizer(dec_prompt_text)
                dec_prompt_ids, dec_prompt_mask_ids = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                assert len(dec_prompt_ids)<=self.config.max_prompt_seq_length, f"\n{example}\n{arg_list}\n{dec_prompt_text}"
                while len(dec_prompt_ids) < self.config.max_prompt_seq_length:
                    dec_prompt_ids.append(self.tokenizer.pad_token_id)
                    dec_prompt_mask_ids.append(self.config.pad_mask_token)
            else:
                raise ValueError(f"no prompt provided for event: {event_type}")
            
            arg_list = self.argument_dict[event_type.replace(':', '.').replace('_', '.')] 
            # NOTE: Large change - original only keep one if multiple span for one arg role
            arg_joint_prompt = dict()
            target_info = dict()
            if os.environ.get("DEBUG", False): arg_set=set()
            for arg in arg_list:
                prompt_slots = None
                arg_target = {
                    "text": list(),
                    "span_s": list(),
                    "span_e": list()
                }

                prompt_slots = {
                    "tok_s":list(), "tok_e":list(),
                }
                
                # Using this more accurate regular expression might further improve rams results
                for matching_result in re.finditer(r'\b'+re.escape(arg)+r'\b', dec_prompt_text.split('.')[0]): 
                    char_idx_s, char_idx_e = matching_result.span(); char_idx_e -= 1
                    tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                    tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                    prompt_slots["tok_s"].append(tok_prompt_s);prompt_slots["tok_e"].append(tok_prompt_e)

                answer_texts, start_positions, end_positions = list(), list(), list()
                if arg in event_args_name:
                    # Deal with multi-occurance
                    if os.environ.get("DEBUG", False): arg_set.add(arg)
                    arg_idxs = [i for i, x in enumerate(event_args_name) if x == arg]
                    if os.environ.get("DEBUG", False): counter[0] += 1; counter[1]+=len(arg_idxs)

                    for arg_idx in arg_idxs:
                        event_arg_info = arguments[arg_idx]
                        answer_texts.append(event_arg_info[3])
                        start_positions.append(old_tok_to_new_tok_index[event_arg_info[0]][0])
                        end_positions.append(old_tok_to_new_tok_index[event_arg_info[1]-1][1])
                        
                arg_target["span_s"]= start_positions
                arg_target["span_e"] = end_positions

                arg_target["text"] = answer_texts
                arg_joint_prompt[arg] = prompt_slots
                target_info[arg] = arg_target
                
            # NOTE: one annotation as one decoding input
            feature_idx = len(features)
            # ipdb.set_trace()
            features.append(
                    InputFeatures(doc_id, feature_idx, 
                                event_type, event_trigger,
                                enc_text, enc_input_ids, enc_mask_ids, 
                                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                                None, arg_joint_prompt, target_info,
                                old_tok_to_new_tok_index = old_tok_to_new_tok_index, full_text=tokens, arg_list = arg_list
                    )
            )

        if os.environ.get("DEBUG", False): print('\033[91m'+f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"+'\033[0m')
        return features
    
    def convert_features_to_batch(self, features):
        # collate_fn

        enc_input_ids = torch.tensor([f.enc_input_ids for f in features]).cuda()
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in features]).cuda()

        dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in features]).cuda()
        dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in features]).cuda()

        target_info = [f.target_info for f in features]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in features]
        arg_joint_prompt = [f.arg_joint_prompt for f in features]
        arg_lists = [f.arg_list for f in features]
        
        return (enc_input_ids, enc_mask_ids, \
                dec_prompt_ids, dec_prompt_mask_ids,\
                target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists)
