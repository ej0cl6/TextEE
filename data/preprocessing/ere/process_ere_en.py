import os, re, json, glob, tqdm, random
import numpy as np
from bs4 import BeautifulSoup
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import stanza
import ipdb

LANG_MAP={
    "english": "en"
}

relation_type_mapping = {
    'orgaffiliation': 'ORG-AFF',
    'personalsocial': 'PER-SOC',
    'physical': 'PHYS',
    'generalaffiliation': 'GEN-AFF',
    'partwhole': 'PART-WHOLE',
}

relation_subtype_mapping = {
    'business': 'Business',
    'employmentmembership': 'EmploymentMembership',
    'family': 'Family',
    'founder': 'Founder',
    'investorshareholder': 'InvestorShareholder',
    'leadership': 'Leadership',
    'locatednear': 'LocatedNear',
    'membership': 'Membership',
    'more': 'MORE',
    'opra': 'OPRA',
    'orgheadquarter': 'OrgHeadquarter',
    'orglocationorigin': 'OrgLocationOrigin',
    'ownership': 'Ownership',
    'resident': 'Resident',
    'studentalum': 'StudentAlum',
    'subsidiary': 'Subsidiary',
    'unspecified': 'Unspecified',
}

event_type_mapping = {
    'business:declarebankruptcy': 'Business:Declare-Bankruptcy',
    'business:endorg': 'Business:End-Org',
    'business:mergeorg': 'Business:Merge-Org',
    'business:startorg': 'Business:Start-Org',
    'conflict:attack': 'Conflict:Attack',
    'conflict:demonstrate': 'Conflict:Demonstrate',
    'contact:broadcast': 'Contact:Broadcast',
    'contact:contact': 'Contact:Contact',
    'contact:correspondence': 'Contact:Correspondence',
    'contact:meet': 'Contact:Meet',
    'justice:acquit': 'Justice:Acquit',
    'justice:appeal': 'Justice:Appeal',
    'justice:arrestjail': 'Justice:Arrest-Jail',
    'justice:chargeindict': 'Justice:Charge-Indict',
    'justice:convict': 'Justice:Convict',
    'justice:execute': 'Justice:Execute',
    'justice:extradite': 'Justice:Extradite',
    'justice:fine': 'Justice:Fine',
    'justice:pardon': 'Justice:Pardon',
    'justice:releaseparole': 'Justice:Release-Parole',
    'justice:sentence': 'Justice:Sentence',
    'justice:sue': 'Justice:Sue',
    'justice:trialhearing': 'Justice:Trial-Hearing',
    'life:beborn': 'Life:Be-Born',
    'life:die': 'Life:Die',
    'life:divorce': 'Life:Divorce',
    'life:injure': 'Life:Injure',
    'life:marry': 'Life:Marry',
    'manufacture:artifact': 'Manufacture:Artifact',
    'movement:transportartifact': 'Movement:Transport-Artifact',
    'movement:transportperson': 'Movement:Transport-Person',
    'personnel:elect': 'Personnel:Elect',
    'personnel:endposition': 'Personnel:End-Position',
    'personnel:nominate': 'Personnel:Nominate',
    'personnel:startposition': 'Personnel:Start-Position',
    'transaction:transaction': 'Transaction:Transaction',
    'transaction:transfermoney': 'Transaction:Transfer-Money',
    'transaction:transferownership': 'Transaction:Transfer-Ownership',
}

role_type_mapping = {
    'adjudicator': 'Adjudicator',
    'agent': 'Agent',
    'artifact': 'Artifact',
    'attacker': 'Attacker',
    'audience': 'Audience',
    'beneficiary': 'Beneficiary',
    'defendant': 'Defendant',
    'destination': 'Destination',
    'entity': 'Entity',
    'giver': 'Giver',
    'instrument': 'Instrument',
    'org': 'Org',
    'origin': 'Origin',
    'person': 'Person',
    'place': 'Place',
    'plaintiff': 'Plaintiff',
    'prosecutor': 'Prosecutor',
    'recipient': 'Recipient',
    'target': 'Target',
    'thing': 'Thing',
    'victim': 'Victim',
}

def recover_escape(text: str) -> str:
    """Converts named character references in the given string to the corresponding
    Unicode characters. I didn't notice any numeric character references in this
    dataset.

    Args:
        text (str): text to unescape.
    
    Returns:
        str: unescaped string.
    """
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('\n', ' ')

@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace('\n', ' ')

    def char_offsets_to_token_offsets(self, tokens):
        """Converts self.start and self.end from character offsets to token
        offsets.

        Args:
            tokens (List[int, int, str]): a list of token tuples. Each item in
                the list is a triple (start_offset, end_offset, text).
        """
        start_ = end_ = -1
        for i, (s, e, _) in enumerate(tokens):
            if s == self.start:
                start_ = i
            if e == self.end:
                end_ = i + 1
        if start_ == -1 or end_ == -1 or start_ > end_:
            raise ValueError('Failed to update offsets for {}-{}:{} in {}'.format(
                self.start, self.end, self.text, tokens))
        self.start, self.end = start_, end_

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            dict: a dict of instance variables.
        """
        return {
            'text': recover_escape(self.text),
            'start': self.start,
            'end': self.end,
        }

    def remove_space(self):
        """Removes heading and trailing spaces in the span text."""
        # heading spaces
        text = self.text.lstrip()
        self.start += len(self.text) - len(text)
        # trailing spaces
        after_text = text.rstrip()
        #self.end = self.start + len(text)
        #self.text = text
        self.end += len(after_text) - len(text)
        self.text = after_text

@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    mention_type: str
    value: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            'text': recover_escape(self.text),
            # 'entity_id': self.entity_id,
            'id': self.mention_id, # mention id as id
            'start': self.start,
            'end': self.end,
            'entity_type': self.entity_type,
            'mention_type': self.mention_type,
        }
        if self.value:
            entity_dict['value'] = self.value
        return entity_dict

@dataclass
class RelationArgument:
    mention_id: str
    role: str
    text: str
    pointed_entity: Entity = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'entity_id': self.mention_id,
            'role': self.role,
            'text': recover_escape(self.text),
            'start': self.pointed_entity.start,
            'end': self.pointed_entity.end
        }

@dataclass
class Relation:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'id': self.relation_id,
            'relation_type': relation_type_mapping[self.relation_type],
            'relation_subtype': relation_subtype_mapping[self.relation_subtype],
            'arguments':[
                self.arg1.to_dict(),
                self.arg2.to_dict()
            ]
        }

@dataclass
class EventArgument:
    mention_id: str
    role: str
    text: str
    pointed_entity: Entity = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'entity_id': self.mention_id,
            'role': role_type_mapping[self.role],
            'text': recover_escape(self.text),
            'start': self.pointed_entity.start,
            'end': self.pointed_entity.end
        }

@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: List[EventArgument]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            # 'evnet_id': self.event_id,
            'id': self.mention_id,
            'event_main_type': self.event_type,
            'event_subtype': self.event_subtype,
            'event_type': event_type_mapping["{}:{}".format(self.event_type, self.event_subtype)],
            'trigger': self.trigger.to_dict(),
            'arguments': [arg.to_dict() for arg in self.arguments],
        }

@dataclass
class Sentence(Span):
    doc_id: str
    sent_id: str
    lang: str
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]
    doc_tokens: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'doc_id': self.doc_id,
            'wnd_id': self.sent_id,
            'text': recover_escape(self.text).replace('\t', ' '),
            'lang': LANG_MAP[self.lang],
            'tokens': [recover_escape(t) for t in self.tokens],
            'entity_mentions': [entity.to_dict() for entity in self.entities],
            'event_mentions': [event.to_dict() for event in self.events],
            'relation_mentions': [relation.to_dict() for relation in self.relations],
        }

@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'doc_id': self.doc_id,
            'sentences': [sent.to_dict() for sent in self.sentences if len(sent.tokens) > 1]
        }

def read_source_file(path,
                     language,
                     sent_map,
                    ) -> List[Tuple[str, int, int]]:
    data = open(path, 'r', encoding='utf-8').read()

    min_offset = max(0, data.find('</HEADLINE>')) # remove sentences in headline since annotations in headline is less consistent
    # For example in PROTESTS AS CITY COLLEGE CLOSES A STUDENT CENTER, no event is annotated.

    chunks = []
    for mat in re.finditer(r"<.*?>", data):
        chunks.append((mat.start(), mat.end()))

    sentences = []
    for index in range(len(chunks)-1):
        if chunks[index+1][0] - chunks[index][1] > 1: # next start - now end
            if chunks[index][1] > min_offset:
                sentences.append((data[(chunks[index][1]):(chunks[index+1][0])], chunks[index][1], chunks[index+1][0]))
                
    # Re-tokenize sentences
    doc_id = path.rsplit("/", 1)[1][:-4]
    try:
        for sent in sentences:
            for s in sent_map[doc_id][f"{sent[1]},{sent[2]}"]:
                continue
    except:
        ipdb.set_trace()
        print()
    sentences_ = [[(sent[0][s[0]-sent[1]:s[1]-sent[1]], s[0], s[1]) for s in sent_map[doc_id][f"{sent[1]},{sent[2]}"]] for sent in sentences]
    sentences_x = [s for sent in sentences_ for s in sent]
    
    return sentences_x

def sent_tokenize(texts, language='english'):
    text, ori_char_start, ori_char_end = texts
    if language == 'english':
        # stanza tokenizer
        doc = nlp_en(text)
        
        ending_char_idx = [0]
        for sent in doc.sentences:
            ending_char_idx.append(sent.tokens[-1].end_char)
        ending_char_idx = ending_char_idx[:-1]
        ending_char_idx.append(len(text))
        sentences = []
        for idx in range(1, len(ending_char_idx)):
            sentences.append(text[ending_char_idx[idx-1]: ending_char_idx[idx]])  

    last = 0
    sentences_ = []
    for sent in sentences:
        index = text[last:].find(sent)
        if index == -1:
            print(text, sent)
        else:
            sentences_.append((sent, last + index + ori_char_start,
                               last + index + len(sent) + ori_char_start))
        last += index + len(sent)
    return sentences_

def read_annotation(path,
                  filler=False,
                  use_full_span=False):
    data = open(path, 'r', encoding='utf-8').read()
    soup = BeautifulSoup(data, 'lxml')

    # metadata
    root = soup.find('deft_ere')
    doc_id = root['doc_id']
    source_type = root['source_type']

    # entities
    entity_list = []
    entities_node = root.find('entities')
    if entities_node:
        for entity_node in entities_node.find_all('entity'):
            entity_id = entity_node['id']
            entity_type = entity_node['type']
            for entity_mention_node in entity_node.find_all('entity_mention'):
                mention_id = entity_mention_node['id']
                mention_type = entity_mention_node['noun_type']
                if (mention_type == 'NOM') and (not use_full_span):
                    mention_offset = int(entity_mention_node.find('nom_head')['offset'])
                    mention_length = int(entity_mention_node.find('nom_head')['length'])
                    mention_text = entity_mention_node.find('nom_head').text
                else:
                    mention_offset = int(entity_mention_node['offset'])
                    mention_length = int(entity_mention_node['length'])
                    mention_text = entity_mention_node.find('mention_text').text
                entity_list.append(Entity(
                    mention_offset, mention_offset + mention_length, mention_text,
                    entity_id, mention_id,
                    entity_type, mention_type,
                ))
    fillers_node = root.find('fillers')
    if fillers_node:
        for filler_node in fillers_node.find_all('filler'):
            entity_id = filler_node['id']
            entity_type = filler_node['type']
            if entity_type == 'weapon':
                entity_type = 'WEA'
            elif entity_type == 'vehicle':
                entity_type = 'VEH'
            else:
                if not filler:
                    continue
            mention_offset = int(filler_node['offset'])
            mention_length = int(filler_node['length'])
            mention_text = filler_node.text
            entity_list.append(
                Entity(
                    mention_offset, mention_offset + mention_length, mention_text,
                    entity_id, entity_id,
                    entity_type, "NOM",
                )
            )

    # relations
    relation_list = []
    relations_node = root.find('relations')
    if relations_node:
        for relation_node in relations_node.find_all('relation'):
            relation_id = relation_node['id']
            relation_type = relation_node['type']
            relation_subtype = relation_node['subtype']
            for relation_mention_node in relation_node.find_all('relation_mention'):
                mention_id = relation_mention_node['id']
                arg1 = relation_mention_node.find('rel_arg1')
                arg2 = relation_mention_node.find('rel_arg2')
                if arg1 and arg2:
                    if arg1.has_attr('entity_id'):
                        arg1_entity_id = arg1['entity_id']
                        arg1_mention_id = arg1['entity_mention_id']
                    else:
                        arg1_entity_id = arg1['filler_id']
                        arg1_mention_id = arg1['filler_id']
                    arg1_role = arg1['role']
                    arg1_text = arg1.text
                    if arg2.has_attr('entity_id'):
                        arg2_entity_id = arg2['entity_id']
                        arg2_mention_id = arg2['entity_mention_id']
                    else:
                        arg2_entity_id = arg2['filler_id']
                        arg2_mention_id = arg2['filler_id']
                    arg2_role = arg2['role']
                    arg2_text = arg2.text
                    relation_list.append(Relation(
                        relation_id=mention_id,
                        relation_type=relation_type,
                        relation_subtype=relation_subtype,
                        arg1=RelationArgument(mention_id=arg1_mention_id,
                                              role=arg1_role,
                                              text=arg1_text),
                        arg2=RelationArgument(mention_id=arg2_mention_id,
                                              role=arg2_role,
                                              text=arg2_text)))

    # events
    event_list = []
    events_node = root.find('hoppers')
    if events_node:
        for event_node in events_node.find_all('hopper'):
            event_arguments = []
            event_id = event_node['id']
            for event_mention_node in event_node.find_all('event_mention'):
                trigger = event_mention_node.find('trigger')
                trigger_offset = int(trigger['offset'])
                trigger_length = int(trigger['length'])
                arguments = []
                for arg in event_mention_node.find_all('em_arg'):
                    if arg['realis'] == 'false':
                        continue
                    if arg.has_attr('entity_id'):
                        arguments.append(EventArgument(
                            mention_id=arg['entity_mention_id'],
                            role=arg['role'],
                            text=arg.text))
                    elif arg.has_attr('filler_id'):
                        arguments.append(EventArgument(
                            mention_id=arg['filler_id'],
                            role=arg['role'],
                            text=arg.text
                        ))
                event_list.append(Event(
                    event_id=event_id,
                    mention_id=event_mention_node['id'],
                    event_type=event_mention_node['type'],
                    event_subtype=event_mention_node['subtype'],
                    trigger=Span(start=trigger_offset,
                                 end=trigger_offset + trigger_length,
                                 text=trigger.text),
                    arguments=arguments))
    
    # remove heading/tailing spaces
    for entity in entity_list:
        entity.remove_space()
    for event in event_list:
        event.trigger.remove_space()

    return (doc_id, source_type, entity_list, relation_list, event_list)

def process_entities(entities,
                     sentences
                    ):
    """
    Cleans entities and splits them into lists
    """
    sentence_entities = [[] for _ in range(len(sentences))]
    cross_boundary_entity_count = 0

    # assign each entity to the sentence where it appears
    for entity in entities:
        start, end = entity.start, entity.end
        flag = False
        for i, (sent_text, s, e) in enumerate(sentences):
            if start >= s and end <= e:
                sentence_entities[i].append(entity)
                flag = True
                # check label and text consistency
                assert entity.text == recover_escape(sent_text[start-s: end-s]), ipdb.set_trace()
                break

            # TODO: Assume no entities that crosses sentence boundary, which could be wrong.
        if not flag:
            cross_boundary_entity_count += 1

    return sentence_entities, cross_boundary_entity_count

def process_relation(relations,
                     sentence_entities,
                     sentences):
    sentence_relations = [[] for _ in range(len(sentences))]
    cross_sentence_relation_count = 0
    for relation in relations:
        mention_id1 = relation.arg1.mention_id
        mention_id2 = relation.arg2.mention_id
        flag = False
        for i, entities in enumerate(sentence_entities):
            arg1_in_sent = any([mention_id1 == e.mention_id for e in entities])
            arg2_in_sent = any([mention_id2 == e.mention_id for e in entities])
            if arg1_in_sent and arg2_in_sent:
                # Ensure the entities are in the same sentence

                # Update relation argument
                entmap = {e.mention_id: e for e in entities}
                sentence_relations[i].append(
                    Relation(relation.relation_id, relation.relation_type,
                    relation.relation_subtype, 
                    RelationArgument(relation.arg1.mention_id, relation.arg1.role, entmap[relation.arg1.mention_id].text, entmap[relation.arg1.mention_id]), 
                    RelationArgument(relation.arg2.mention_id, relation.arg2.role, entmap[relation.arg2.mention_id].text, entmap[relation.arg2.mention_id]), )
                )
                flag = True
                break
        if not flag:
            cross_sentence_relation_count += 1
    return sentence_relations, cross_sentence_relation_count

def process_events(events,
                   sentence_entities,
                   sentences
                  ):
    
    sentence_events = [[] for _ in range(len(sentences))]
    # assign each event mention to the sentence where it appears
    cross_boundary_trigger_count = 0
    cross_boundary_argument_count = 0
    for event in events:
        start, end = event.trigger.start, event.trigger.end
        flag_trigger = False
        for i, (sent_text, s, e) in enumerate(sentences):
            # check trigger
            # TODO: Here, we assume no cross single sentence boundary trigger
            if start >= s and end <= e:
                flag_trigger = True

                # check label and text consistency
                assert event.trigger.text == recover_escape(sent_text[start-s: end-s]), ipdb.set_trace()

                sent_entities = sentence_entities[i]
                # clean the argument list
                arguments = []
                for argument in event.arguments:
                    flag_argument = False
                    mention_id = argument.mention_id
                    for entity in sent_entities:
                        if entity.mention_id == mention_id:
                            arguments.append(EventArgument(argument.mention_id, argument.role, entity.text, entity))
                            flag_argument = True
                            break
                    if not flag_argument:
                        if '-E' in argument.mention_id:
                            cross_boundary_argument_count += 1
                event_cleaned = Event(event.event_id, event.mention_id,
                                    event.event_type, event.event_subtype,
                                    trigger=event.trigger,
                                    arguments=arguments)
                sentence_events[i].append(event_cleaned)
                break # no need to iterate sentences
        if not flag_trigger:
            cross_boundary_trigger_count += 1
    return sentence_events, cross_boundary_trigger_count, cross_boundary_argument_count

def tokenize(sentence,
             entities,
             events,
             doc_id,
             token_map,
             language):
    text, start, end = sentence
    tokens = [(t[0], t[1], text[t[0]-start:t[1]-start]) for t in token_map[doc_id][f"{start},{end}"]]
                
    return tokens

def convert(source_file,
            annotation_file,
            sent_map,
            token_map,
            filler,
            language,
            use_full_span):

    sentences = read_source_file(source_file, language, sent_map)
    # sentences is a list of tuple (sentence, start_char_index, end_char_index)
    
    doc_id, source_type, entities, relations, events = read_annotation(annotation_file, filler=filler, use_full_span=use_full_span)

    # Process entities, relations, and events
    sentence_entities, cross_boundary_entity_count = process_entities(entities, sentences)
    sentence_relations, cross_boundary_relation_count = process_relation(
        relations, sentence_entities, sentences)
    sentence_events, cross_boundary_trigger_count, cross_boundary_argument_count = process_events(events, sentence_entities, sentences)

    #####
    # Right now, all the annotation is under characater span level.
    #####

    # Tokenization
    sentence_tokens = [tokenize(s, ent, evt, doc_id, token_map, language=language) for s, ent, evt
                       in zip(sentences, sentence_entities, sentence_events)]

    # Convert span character offsets to token indices
    sentence_objs = []
    doc_tokens = [tok for sent in sentence_tokens for tok in sent]
    for i, (toks, ents, evts, rels, sent) in enumerate(zip(
            sentence_tokens, sentence_entities, sentence_events,
            sentence_relations, sentences)):
        for entity in ents:
            entity.char_offsets_to_token_offsets(toks)
        for event in evts:
            event.trigger.char_offsets_to_token_offsets(toks)
        wnd_id = '{}-{}'.format(doc_id, i)
        sent_obj = Sentence(start=sent[1],
                            end=sent[2],
                            text=sent[0],
                            doc_id=doc_id,
                            sent_id=wnd_id,
                            lang=language,
                            tokens=[t for _, _, t in toks],
                            entities=ents,
                            relations=rels,
                            events=evts,
                            doc_tokens=[tok for sent in sentence_tokens for _,_, tok in sent])
        sent_obj.remove_space()
        try:
            sent_obj.char_offsets_to_token_offsets(doc_tokens)
        except:
            ipdb.set_trace()
        sentence_objs.append(sent_obj)
    return Document(doc_id, sentence_objs), cross_boundary_entity_count, cross_boundary_relation_count,cross_boundary_trigger_count, cross_boundary_argument_count 

def convert_batch(input_path,
                  sent_map, 
                  token_map, 
                  language='english',
                  filler=False,
                  use_full_span=False):
    
    # Read all file
    if language == 'english':
        annotation_files = glob.glob(os.path.join(
            input_path, 'rich_ere', '**', '*.rich_ere.xml'))
        # get source file
        source_files = [(fname.split('rich_ere/')[-1]).split('.rich_ere.xml')[0] for fname in annotation_files]
        source_files = [os.path.join(
            input_path, 'source', '{}.cmp.txt'.format(fname)) if 'discussion_forum' in fname else os.path.join(
            input_path, 'source', '{}.txt'.format(fname)) for fname in source_files]
    else:
        raise ValueError('Unknown language: {}'.format(language))
    print(input_path)
    print('#Rich ERE files: {}'.format(len(source_files)))
    
    progress = tqdm.tqdm(total=len(source_files))
    cross_boundary_entity_counts, cross_boundary_relation_counts, cross_boundary_trigger_counts, cross_boundary_argument_counts = 0, 0, 0, 0

    objs = []
    for source_file, annotation_file in zip(source_files, annotation_files):
        progress.update(1)
        doc, cross_boundary_entity_count, cross_boundary_relation_count, cross_boundary_trigger_count, cross_boundary_argument_count = convert(source_file, annotation_file, sent_map, token_map, filler=filler, language=language, use_full_span=use_full_span)
        cross_boundary_argument_counts += cross_boundary_argument_count
        cross_boundary_entity_counts += cross_boundary_entity_count
        cross_boundary_trigger_counts += cross_boundary_trigger_count
        cross_boundary_relation_counts += cross_boundary_relation_count
        if doc:
            objs.append(doc.to_dict())

    print('Cross Boundary Entity: {}'.format(cross_boundary_entity_counts))
    print('Cross Boundary Relation: {}'.format(cross_boundary_relation_counts))
    print('Cross Boundary Trigger: {}'.format(cross_boundary_trigger_counts))
    print('Cross Boundary Argument: {}'.format(cross_boundary_argument_counts))
    progress.close()
    
    return objs

def get_split(objs, split_path, split_folder):
    
    with open(os.path.join(split_path, split_folder, "train.txt")) as fp:
        lines = fp.readlines()
        train_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split_path, split_folder, "dev.txt")) as fp:
        lines = fp.readlines()
        dev_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split_path, split_folder, "test.txt")) as fp:
        lines = fp.readlines()
        test_doc_ids = set([l.strip() for l in lines])

    train_objs = []
    dev_objs = []
    test_objs = []
    
    for obj in objs:
        if obj["doc_id"] in train_doc_ids:
            train_objs.extend(obj['sentences'])
        elif obj["doc_id"] in dev_doc_ids:
            dev_objs.extend(obj['sentences'])
        elif obj["doc_id"] in test_doc_ids:
            test_objs.extend(obj['sentences'])
        else:
            # ipdb.set_trace()
            raise ValueError("split docs are not complete")
    
    train_objs.sort(key=lambda x: x["doc_id"])
    dev_objs.sort(key=lambda x: x["doc_id"])
    test_objs.sort(key=lambda x: x["doc_id"])
    
    return train_objs, dev_objs, test_objs

def get_statistics(data):
    event_type_count = defaultdict(int)
    role_type_count = defaultdict(int)
    relation_type_count = defaultdict(int)
    relation_subtype_count = defaultdict(int)
    entity_type_count = defaultdict(int)
    doc_ids = set()
    max_len = 0
    for dt in data:
        max_len = max(max_len, len(dt["tokens"]))
        doc_ids.add(dt["doc_id"])
        for event in dt["event_mentions"]:
            event_type_count[event["event_type"]] += 1
            for argument in event["arguments"]:
                role_type_count[argument["role"]] += 1
        for ent in dt['entity_mentions']:
            entity_type_count[ent['entity_type']] += 1
        for rel in dt['relation_mentions']:
            relation_type_count[rel['relation_type']] += 1
            relation_subtype_count[rel['relation_subtype']] += 1
    
    print(f"# of Instances: {len(data)}")
    print(f"# of Docs: {len(doc_ids)}")
    print(f"Max Length: {max_len}")
    print(f"# of Event Types: {len(event_type_count)}")
    print(f"# of Events: {sum(event_type_count.values())}")
    print(f"# of Role Types: {len(role_type_count)}")
    print(f"# of Arguments: {sum(role_type_count.values())}")
    # pprint(event_type_count)
    print()

def save_data(out_path, split, train_data, dev_data, test_data):
    data_path = os.path.join(out_path, f"{split}")
    os.makedirs(data_path)
    
    with open(os.path.join(data_path, "train.json"), "w") as fp:
        for data in train_data:
            fp.write(json.dumps(data)+"\n")
    
    with open(os.path.join(data_path, "dev.json"), "w") as fp:
        for data in dev_data:
            fp.write(json.dumps(data)+"\n")
    
    with open(os.path.join(data_path, "test.json"), "w") as fp:
        for data in test_data:
            fp.write(json.dumps(data)+"\n")

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--in_folder", help="Path to the input folder")
    parser.add_argument("-o", "--out_folder", help="Path to the output folder")
    parser.add_argument('-l', '--lang', default='english',
                        choices=['english'],
                        help='Document language')
    parser.add_argument('--filler', action='store_true',
                        help='Extracts times, money, crime, etc. (None named-entity arguments)')
    parser.add_argument('--use_full_span', action='store_true',
                        help='Use full span instead of head span for named-entities')
    parser.add_argument('--split_path', help="Path to split folder")
    parser.add_argument('--split', help="Path to split folder")
    parser.add_argument('--sent_map', default='sent_map.json')
    parser.add_argument('--token_map', default='token_map.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    with open(args.sent_map) as fp:
        sent_map = json.load(fp)
    with open(args.token_map) as fp:
        token_map = json.load(fp)
    objs = convert_batch(args.in_folder, sent_map, token_map, language=args.lang, filler=args.filler, use_full_span=args.use_full_span)
    
    train_objs, dev_objs, test_objs = get_split(objs, args.split_path, args.split)
    
    print("Train")
    get_statistics(train_objs)
    print("Dev")
    get_statistics(dev_objs)
    print("Test")
    get_statistics(test_objs)
    
    save_data(args.out_folder, args.split, train_objs, dev_objs, test_objs)

if __name__ == '__main__':
    main()