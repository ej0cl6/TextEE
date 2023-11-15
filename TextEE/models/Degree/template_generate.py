from .pattern import patterns, ROLE_PH_MAP
import re
import ipdb

INPUT_STYLE_SET = ['event_type', 'event_type_sent', 'keywords', 'triggers', 'template']
OUTPUT_STYLE_SET = ['trigger:sentence', 'argument:sentence']
TRIGGER_PH_MAP = {
    'Trigger': '<Trigger>'
}
ROLE_TEMPLATE_PREFIX = "ROLE_"
SEP = "\n"
AND = "and"
PAD = "<pad>"

class eve_template_generator():
    def __init__(self, dataset, passage, triggers, roles, input_style, output_style, vocab, instance_base=False):
        """
        generate strctured information for events
        
        args:
            dataset(str): which dataset is used
            passage(List): a list of tokens
            triggers(List): a list of triggers
            roles(List): a list of Roles
            input_style(List): List of elements; elements belongs to INPUT_STYLE_SET
            input_style(List): List of elements; elements belongs to OUTPUT_STYLE_SET
            instance_base(Bool): if instance_base, we generate only one pair (use for trigger generation), else, we generate trigger_base (use for argument generation)
        """
        self.raw_passage = passage
        self.triggers = triggers
        self.roles = roles
        self.events = self.process_events(passage, triggers, roles)
        self.input_style = input_style
        self.output_style = output_style
        self.vocab = vocab
        self.event_templates = []
        if instance_base:
            for e_type in self.vocab['event_type_itos']:
                self.event_templates.append(
                    event_template(e_type, patterns[dataset][e_type], 
                    self.input_style, self.output_style, passage, ROLE_PH_MAP[dataset], self.events)
                )
        else:
            for event in self.events:
                self.event_templates.append(
                    event_template(event['event type'], patterns[dataset][event['event type']], 
                    self.input_style, self.output_style, event['tokens'], ROLE_PH_MAP[dataset], event)
                )
        self.data = [x.generate_pair(x.trigger_text) for x in self.event_templates]
        self.data = [x for x in self.data if x]

    def get_training_data(self):
        return self.data

    def process_events(self, passage, triggers, roles):
        """
        Given a list of token and event annotation, return a list of structured event

        structured_event:
        {
            'trigger text': str,
            'trigger span': (start, end),
            'event type': EVENT_TYPE(str),
            'arguments':{
                ROLE_TYPE(str):[{
                    'argument text': str,
                    'argument span': (start, end)
                }],
                ROLE_TYPE(str):...,
                ROLE_TYPE(str):....
            }
            'passage': PASSAGE
        }
        """
        
        events = {trigger: [] for trigger in triggers}

        for argument in roles:
            trigger = argument[0]
            events[trigger].append(argument)
        
        event_structures = []
        for trigger, arguments in events.items():
            eve_type = trigger[2]
            eve_text = ' '.join(passage[trigger[0]:trigger[1]])
            eve_span = (trigger[0], trigger[1])
            argus = {}
            for argument in arguments:
                role_type = argument[1][2]
                if role_type not in argus.keys():
                    argus[role_type] = []
                argus[role_type].append({
                    'argument text': ' '.join(passage[argument[1][0]:argument[1][1]]),
                    'argument span': (argument[1][0], argument[1][1]),
                })
            event_structures.append({
                'trigger text': eve_text,
                'trigger span': eve_span,
                'event type': eve_type,
                'arguments': argus,
                'passage': ' '.join(passage),
                'tokens': passage
            })
        return event_structures

def format_template(template, ROLE_PH_MAP):
    return template.format(**{**ROLE_PH_MAP, **TRIGGER_PH_MAP})

class event_template():
    def __init__(self, event_type, info_dict, input_style, output_style, passage, ROLE_PH_MAP, gold_event=None):
        self.ROLE_PH_MAP = ROLE_PH_MAP
        self.info_dict = info_dict
        self.event_type = event_type
        self.input_style = input_style
        self.output_style = output_style
        self.output_template = self.get_output_template()
        self.passage = ' '.join(passage) # Assume this is English
        self.tokens = passage
        
        if gold_event is not None:
            self.gold_event = gold_event
            if isinstance(gold_event, list):
                # instance base
                self.trigger_text = f" {AND} ".join([x['trigger text'] for x in gold_event if x['event type']==event_type])
                self.trigger_span = [x['trigger span'] for x in gold_event if x['event type']==event_type]
                self.arguments = [x['arguments'] for x in gold_event if x['event type']==event_type]
            else:
                # trigger base
                self.trigger_text = gold_event['trigger text']
                self.trigger_span = [gold_event['trigger span']]
                self.arguments = [gold_event['arguments']]         
        else:
            self.gold_event = None
        
    def get_keywords(self):
        return self.info_dict['keywords']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' {} {}'.format(SEP, format_template(self.info_dict['ED template'], self.ROLE_PH_MAP))
                if o_style == 'argument:sentence':
                    output_template += ' {} {}'.format(SEP, format_template(self.info_dict['EAE template'], self.ROLE_PH_MAP))
        return (f'{SEP}'.join(output_template.split(f'{SEP}')[1:])).strip()

    def generate_pair(self, query_trigger):
        """
        Generate model input sentence and output sentence pair
        """
        input_str, supplements = self.generate_input_str_detail(query_trigger)
        output_str, gold_sample = self.generate_output_str(query_trigger)
        return (input_str, output_str, self.gold_event, gold_sample, self.event_type, self.tokens, supplements)

    def generate_input_str_detail(self, query_trigger):
        input_str = ''
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' {} {}'.format(SEP, self.info_dict['event type'])
                if i_style == 'event_type_sent':
                    input_str += ' {} {}'.format(SEP, self.info_dict['event description'])
                if i_style == 'keywords':
                    input_str += ' {} Similar triggers such as {}'.format(SEP, ', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' {} The event trigger word is {}'.format(SEP, query_trigger)
                if i_style == 'template':
                    input_str += ' {} {}'.format(SEP, self.output_template)
        return self.passage+input_str, input_str

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' {} {}'.format(SEP, self.info_dict['event type'])
                if i_style == 'event_type_sent':
                    input_str += ' {} {}'.format(SEP, self.info_dict['event description'])
                if i_style == 'keywords':
                    input_str += ' {} Similar triggers such as {}'.format(SEP, ', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' {} The event trigger word is {}'.format(SEP, query_trigger)
                if i_style == 'template':
                    input_str += ' {} {}'.format(SEP, self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    filler = dict()
                    if self.trigger_text != '':
                        filler["Trigger"] = self.trigger_text
                        gold_sample = True
                    else:
                        filler["Trigger"] = TRIGGER_PH_MAP['Trigger']
                    output_str += ' {} {}'.format(SEP, self.info_dict['ED template'].format(**filler))

                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = dict()
                        roles = re.findall(r"{[^/}][^}]*}", self.info_dict['EAE template'])
                        roles = [role[1:-1].split(ROLE_TEMPLATE_PREFIX, 1)[1] for role in roles]
                        for role_type in roles:
                            filler['{}{}'.format(ROLE_TEMPLATE_PREFIX, role_type)] = f" {AND} ".join([ a['argument text'] for a in argu[role_type]]) if role_type in argu.keys() else self.ROLE_PH_MAP['ROLE_{}'.format(role_type)]
                        output_texts.append(self.info_dict['EAE template'].format(**filler))
                        gold_sample = True
                    output_str += ' {} {}'.format(SEP, ' <sep> '.join(output_texts))

        output_str = (f'{SEP}'.join(output_str.split(f'{SEP}')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split(f'{SEP}')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            # try:
                            #     contexts = re.split(r"{[^/}][^}]*}", self.info_dict['ED template'])
                            #     triggers = []
                            #     for idx in range(len(contexts)-1):
                            #         trigger = full_pred.split(contexts[idx], 1)[1]
                            #         trigger = trigger.split(contexts[idx+1], 1)[0]
                            #         triggers.append(trigger.strip())
                            #     triggers = [tri for trigger in triggers for tri in trigger.split(' and ') ]
                            #     for t_cnt, t in enumerate(triggers):
                            #         if t != TRIGGER_PH_MAP['Trigger'] and t != '':
                            #             output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            # except:
                            #     pass
                            contexts = re.split(r"{[^/}][^}]*}", self.info_dict['ED template'])
                            triggers = []
                            for idx in range(len(contexts)-1):
                                try:
                                    trigger = full_pred.split(contexts[idx], 1)[1]
                                    trigger = trigger.split(contexts[idx+1], 1)[0]
                                    triggers.append(trigger.strip())
                                except:
                                    pass
                            triggers = [tri for trigger in triggers for tri in trigger.split(f' {AND} ')]
                            for t_cnt, t in enumerate(triggers):
                                if t != TRIGGER_PH_MAP['Trigger'] and t != '':
                                    output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                contexts = re.split(r"{[^/}][^}]*}", self.info_dict['EAE template'])
                                roles = re.findall(r"{[^/}][^}]*}", self.info_dict['EAE template'])
                                roles = [role[1:-1].split(ROLE_TEMPLATE_PREFIX, 1)[1] for role in roles]
                                assert len(contexts) == len(roles)+1

                                for idx in range(len(contexts)-1):
                                    try:
                                        if contexts[idx] != '':
                                            pred_argu = prediction.split(contexts[idx], 1)[1]
                                        else:
                                            pred_argu = prediction
                                        if contexts[idx+1] != '':
                                            pred_argu = pred_argu.split(contexts[idx+1], 1)[0]
                                        pred_argu = pred_argu.split(f' {AND} ')
                                        for argu in pred_argu:
                                            if argu != self.ROLE_PH_MAP["{}{}".format(ROLE_TEMPLATE_PREFIX, roles[idx])]:
                                                if argu != '':
                                                    output.append((argu, roles[idx], {'cor tri cnt': a_cnt}))
                                    except:
                                        pass
                        used_o_cnt += 1
                    
        return output

    def evaluate(self, predict_output):
        assert self.gold_event is not None
        # categorize prediction
        pred_trigger = []
        pred_argument = []
        for pred in predict_output:
            if pred[1] == self.event_type:
                pred_trigger.append(pred)
            else:
                pred_argument.append(pred)
        
        # get trigger id map
        pred_trigger_map = {}
        for p_tri in pred_trigger:
            # assert p_tri[2]['tri counter'] not in pred_trigger_map.keys()
            pred_trigger_map[p_tri[2]['tri counter']] = p_tri

        # trigger score
        gold_tri_num = len(self.trigger_span)
        pred_tris = []
        for pred in pred_trigger:
            pred_span = self.predstr2span(pred[0])
            if pred_span[0] > -1:
                pred_tris.append((pred_span[0], pred_span[1], pred[1]))
        pred_tri_num = len(pred_tris)
        match_tri = 0
        for pred in pred_tris:
            id_flag = False
            for gold_span in self.trigger_span:
                if gold_span[0] == pred[0] and gold_span[1] == pred[1]:
                    id_flag = True
            match_tri += int(id_flag)

        # argument score
        converted_gold = self.get_converted_gold()
        gold_arg_num = len(converted_gold)
        pred_arg = []
        for pred in pred_argument:
            # find corresponding trigger
            pred_span = None
            if isinstance(self.gold_event, list):
                # end2end case
                try:
                    # we need this ``try'' because we cannot gurantee the model will be bug-free on the matching
                    cor_tri = pred_trigger_map[pred[2]['cor tri cnt']]
                    cor_tri_span_head = self.predstr2span(cor_tri[0])[0]
                    if cor_tri_span_head > -1:
                        pred_span = self.predstr2span(pred[0], cor_tri_span_head)
                    else:
                        continue
                except Exception as e:
                    print('unmatch exception')
                    print(e)
            else:
                # argument only case
                pred_span = self.predstr2span(pred[0], self.trigger_span[0][0])
            if (pred_span is not None) and (pred_span[0] > -1):
                pred_arg.append((pred_span[0], pred_span[1], pred[1]))
        pred_arg = list(set(pred_arg))
        pred_arg_num = len(pred_arg)
        
        target = converted_gold
        match_id = 0
        match_type = 0
        for pred in pred_arg:
            id_flag = False
            id_type = False
            for gold in target:
                if gold[0]==pred[0] and gold[1]==pred[1]:
                    id_flag = True
                    if gold[2] == pred[2]:
                        id_type = True
                        break
            match_id += int(id_flag)
            match_type += int(id_type)
        return {
            'gold_tri_num': gold_tri_num, 
            'pred_tri_num': pred_tri_num,
            'match_tri_num': match_tri,
            'gold_arg_num': gold_arg_num,
            'pred_arg_num': pred_arg_num,
            'match_arg_id': match_id,
            'match_arg_cls': match_type
        }
    
    def get_converted_gold(self):
        converted_gold = []
        for argu in self.arguments:
            for arg_type, arg_list in argu.items():
                for arg in arg_list:
                    converted_gold.append((arg['argument span'][0], arg['argument span'][1], arg_type))
        return list(set(converted_gold))
    
    def predstr2span(self, pred_str, trigger_idx=None):
        sub_words = [_.strip() for _ in pred_str.strip().lower().split()]
        candidates=[]
        for i in range(len(self.tokens)):
            j = 0
            while j < len(sub_words) and i+j < len(self.tokens):
                if self.tokens[i+j].lower() == sub_words[j]:
                    j += 1
                else:
                    break
            if j == len(sub_words):
                candidates.append((i, i+len(sub_words)))
        if len(candidates) < 1:
            return -1, -1
        else:
            if trigger_idx is not None:
                return sorted(candidates, key=lambda x: abs(trigger_idx-x[0]))[0]
            else:
                return candidates[0]
