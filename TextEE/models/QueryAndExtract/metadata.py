import json
import ipdb

class Metadata(object):
    def __init__(self, metadata_path, dataset, type_set):
        self.pos_set = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                        'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        self.pos2id = dict((v, i) for v, i in zip(sorted(self.pos_set), range(len(self.pos_set))))
        self.entity_to_ids = {'FAC': 0, 'GPE': 1, 'LOC': 2, 'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6, 'O': 7, '[PAD]': 8}

        with open(metadata_path, 'r') as j:
            meta = json.loads(j.read())
        self.dataset = dataset
        self.type_set = type_set
        self.metadata = DatasetFactContainer(meta[dataset], type_set=type_set)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


class DatasetFactContainer(object):
    def __init__(self, trigger_arg_dic, arg_entity_dic=None, type_set=None):
        self.trigger_arg_dic = trigger_arg_dic
        self.arg_entity_dic = arg_entity_dic
        self.type_set = type_set
        self.trigger_set = {}
        self.arg_set = {}
        self.entity_set = {}
        self.args_to_ids, self.ids_to_args = {}, {}
        self.triggers_to_ids, self.ids_to_triggers = {}, {}
        self.init_setup()

    def init_setup(self):
        if self.type_set == None: # use full dict
            self.trigger_set = set(self.trigger_arg_dic.keys())
            self.arg_set = set(sum(list(self.trigger_arg_dic.values()), []))
        else: # use the passed in type_set
            self.trigger_set = set(self.type_set['trigger'])
            try:
                self.arg_set = set(self.type_set['role'])
            except KeyError: # the case for ED
                self.arg_set = set(sum(list(self.trigger_arg_dic.values()), []))
        self.triggers_to_ids, self.ids_to_triggers = self.set_to_ids(self.trigger_set)
        self.args_to_ids, self.ids_to_args = self.set_to_ids(self.arg_set)

    @staticmethod
    def set_to_ids(input_set):
        items_to_ids = dict((v, i) for v, i in zip(sorted(list(input_set)), range(len(input_set))))
        items_to_ids['O'] = len(items_to_ids)
        items_to_ids['[PAD]'] = len(items_to_ids)
        ids_to_items = dict((v, k) for k, v in items_to_ids.items())
        return items_to_ids, ids_to_items

