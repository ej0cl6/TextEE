import os, logging, json, _jsonnet, random, datetime, pprint
import numpy as np
import torch
from argparse import Namespace
from models import *
import ipdb

logger = logging.getLogger(__name__)

VALID_TASKS = ["E2E", "ED", "EAE", "EARL"]

TRAINER_MAP = {
    ("DyGIEpp", "E2E"): DyGIEppE2ETrainer,
    ("OneIE", "E2E"): OneIEE2ETrainer,
    ("CRFTagging", "ED"): CRFTaggingEDTrainer, 
    ("CRFTagging", "EAE"): CRFTaggingEAETrainer, 
    ("EEQA", "ED"): EEQAEDTrainer, 
    ("EEQA", "EAE"): EEQAEAETrainer, 
    ("RCEE", "ED"): RCEEEDTrainer, 
    ("RCEE", "EAE"): RCEEEAETrainer, 
    ("TagPrime", "ED"): TagPrimeEDTrainer, 
    ("TagPrime", "EAE"): TagPrimeEAETrainer, 
    ("QueryAndExtract", "ED"): QueryAndExtractEDTrainer,
    ("QueryAndExtract", "EAE"): QueryAndExtractEAETrainer,
    ("Degree", "E2E"): DegreeE2ETrainer,
    ("Degree", "ED"): DegreeEDTrainer,
    ("Degree", "EAE"): DegreeEAETrainer,
    ("UniST", "ED"): UniSTEDTrainer,
    ("CEDAR", "ED"): CEDAREDTrainer,
    ("PAIE", "EAE"): PAIEEAETrainer, 
    ("XGear", "EAE"): XGearEAETrainer,
    ("BartGen", "EAE"): BartGenEAETrainer,
    ("Ampere", "EAE"): AmpereEAETrainer,
    ("AMRIE", "E2E"): AMRIEE2ETrainer,
}

def load_config(config_fn):
    config = json.loads(_jsonnet.evaluate_file(config_fn))
    config = Namespace(**config)
    assert config.task in VALID_TASKS, f"Task must be in {VALID_TASKS}"
    
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

def set_gpu(gpu_device):
    if gpu_device >= 0:
        torch.cuda.set_device(gpu_device)
        
def set_logger(config):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
    output_dir = os.path.join(config.output_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_path = os.path.join(output_dir, "train.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                        handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
    
    # save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
        json.dump(vars(config), fp, indent=4)
        
    config.output_dir = output_dir
    config.log_path = log_path
    
    return config

def parse_unknown_args(unknown_args):
    args = {}
    key = None
    for unknown_arg in unknown_args:
        if unknown_arg.startswith("--"):
            key = unknown_arg[2:]
        else:
            args[key] = unknown_arg
    return args

def load_data(task, file, add_extra_info_fn, config):
    
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    objs = [json.loads(line) for line in lines]
    
    if task == "E2E":
        data, type_set = load_E2E_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(data), len(type_set["trigger"]), len(type_set["role"]), file))
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "ED":
        data, type_set = load_ED_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(data), len(type_set["trigger"]), file))
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif task == "EAE":
        data, type_set = load_EAE_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(data), len(type_set["trigger"]), len(type_set["role"]), file))
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "EARL":
        data, type_set = load_EARL_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(data), len(type_set["trigger"]), len(type_set["role"]), file))
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return data, type_set

def load_all_data(config, add_extra_info_fn):
    
    with open(config.train_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    train_objs = [json.loads(line) for line in lines]
    
    with open(config.dev_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    dev_objs = [json.loads(line) for line in lines]
    
    with open(config.test_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    test_objs = [json.loads(line) for line in lines]
    
    if config.task == "E2E":
        train_data, train_type_set = load_E2E_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), len(train_type_set["role"]), config.train_file))
        dev_data, dev_type_set = load_E2E_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), len(dev_type_set["role"]), config.dev_file))
        test_data, test_type_set = load_E2E_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), len(test_type_set["role"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "ED":
        train_data, train_type_set = load_ED_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), config.train_file))
        dev_data, dev_type_set = load_ED_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), config.dev_file))
        test_data, test_type_set = load_ED_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"]}
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif config.task == "EAE":
        train_data, train_type_set = load_EAE_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), len(train_type_set["role"]), config.train_file))
        dev_data, dev_type_set = load_EAE_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), len(dev_type_set["role"]), config.dev_file))
        test_data, test_type_set = load_EAE_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), len(test_type_set["role"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "EARL":
        train_data, train_type_set = load_EARL_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), len(train_type_set["role"]), config.train_file))
        dev_data, dev_type_set = load_EARL_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), len(dev_type_set["role"]), config.dev_file))
        test_data, test_type_set = load_EARL_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), len(test_type_set["role"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return train_data, dev_data, test_data, type_set

def load_text(task, file, add_extra_info_fn, config):
    
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        
    import stanza
    nlp_en = stanza.Pipeline(lang='en', processors='tokenize')
    
    objs = []
    offset_map = []
    for i, line in enumerate(lines):
        line = line.strip()
        res_text = nlp_en(line)
        text_tokens = [t.text for s in res_text.sentences for t in s.tokens]
        text_offsets = [(t.start_char, t.end_char) for s in res_text.sentences for t in s.tokens]
        obj = {
            "doc_id": f"DOC_{i:06d}", 
            "wnd_id": f"DOC_{i:06d}", 
            "text": line, 
            "lang": "en", 
            "tokens": text_tokens,
            "entity_mentions": [], 
            "event_mentions": [], 
        }
        objs.append(obj)
        offset_map.append(text_offsets)

    if task == "E2E":
        data, _ = load_E2E_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances from {}'.format(len(data), file))
    elif task == "ED":
        data, _ = load_ED_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances from {}'.format(len(data), file))
        
    assert len(data) == len(offset_map)
        
    return data, offset_map

def load_E2E_data(data, add_extra_info_fn, config):
    
    instances = []
    for dt in data:

        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        events = []
        entity_map = {entity['id']: entity for entity in entities}
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)

            arguments.sort(key=lambda x: (x[0], x[1]))
            events.append({"trigger": trigger, "arguments": arguments})

        events.sort(key=lambda x: (x['trigger'][0], x['trigger'][1]))
        
        instance = {"doc_id": dt["doc_id"], 
                    "wnd_id": dt["wnd_id"], 
                    "tokens": dt["tokens"], 
                    "text": dt["text"], 
                    "events": events, 
                   }

        instances.append(instance)

    trigger_type_set = set()
    for instance in instances:
        for event in instance['events']:
            trigger_type_set.add(event['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for event in instance['events']:
            for argument in event["arguments"]:
                role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def load_ED_data(data, add_extra_info_fn, config):

    instances = []
    for dt in data:

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        triggers = []
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            triggers.append(trigger)

        triggers.sort(key=lambda x: (x[0], x[1]))
        
        instance = {"doc_id": dt["doc_id"], 
                    "wnd_id": dt["wnd_id"], 
                    "tokens": dt["tokens"], 
                    "text": dt["text"], 
                    "triggers": triggers,
                   }

        instances.append(instance)

    trigger_type_set = set()
    for instance in instances:
        for trigger in instance['triggers']:
            trigger_type_set.add(trigger[2])

    type_set = {"trigger": trigger_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def load_EAE_data(data, add_extra_info_fn, config):

    instances = []
    for dt in data:
        
        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        entity_map = {entity['id']: entity for entity in entities}
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)

            arguments.sort(key=lambda x: (x[0], x[1]))
            
            instance = {"doc_id": dt["doc_id"], 
                        "wnd_id": dt["wnd_id"], 
                        "tokens": dt["tokens"], 
                        "text": dt["text"], 
                        "trigger": trigger, 
                        "arguments": arguments, 
                       }

            instances.append(instance)
            
    trigger_type_set = set()
    for instance in instances:
        trigger_type_set.add(instance['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for argument in instance["arguments"]:
            role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def load_EARL_data(data, add_extra_info_fn, config):

    instances = []
    for dt in data:
        
        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])
        
        entity_map = {entity['id']: entity for entity in entities}

        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])
            
            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)
            
            labeled_entities = set([(a[0], a[1]) for a in arguments]) 
            non_labeled_entities = set([(e['start'], e['end'], None, e['text']) for e in entities if (e['start'], e['end']) not in labeled_entities])
            arguments.extend(list(non_labeled_entities))
            arguments.sort(key=lambda x: (x[0], x[1]))
            
            instance = {"doc_id": dt["doc_id"], 
                        "wnd_id": dt["wnd_id"], 
                        "tokens": dt["tokens"], 
                        "text": dt["text"], 
                        "trigger": trigger, 
                        "arguments": arguments,
                       }

            instances.append(instance)
            
    trigger_type_set = set()
    for instance in instances:
        trigger_type_set.add(instance['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for argument in instance["arguments"]:
            if argument[2] is not None:
                role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def convert_ED_to_EAE(data, gold):
    instances = []
    for dt, gd in zip(data, gold):
        for trigger in dt["triggers"]:
            trigger_ = (trigger[0], trigger[1], trigger[2], " ".join(gd["tokens"][trigger[0]:trigger[1]]))
            instance = {"doc_id": gd["doc_id"], 
                        "wnd_id": gd["wnd_id"], 
                        "tokens": gd["tokens"], 
                        "text": gd["text"], 
                        "trigger": trigger_, 
                        "arguments": [], 
                        "extra_info": gd["extra_info"]
                       }
            instances.append(instance)
    
    return instances

def combine_ED_and_EAE_to_E2E(ed_predicitons, eae_predictions):
    e2e_predictions = []
    idx = 0
    for ed_prediciton in ed_predicitons:
        events = []
        for trigger in ed_prediciton["triggers"]:
            eae_prediction = eae_predictions[idx]
            assert ed_prediciton["doc_id"] == eae_prediction["doc_id"]
            assert ed_prediciton["wnd_id"] == eae_prediction["wnd_id"]
            assert trigger[0] == eae_prediction["trigger"][0]
            assert trigger[1] == eae_prediction["trigger"][1]
            assert trigger[2] == eae_prediction["trigger"][2]
            events.append({"trigger": trigger, "arguments": eae_prediction["arguments"]})
            idx += 1
        
        ed_prediciton["events"] = events
        e2e_predictions.append(ed_prediciton)

    return e2e_predictions

def save_predictions(file, predictions, data=None, offset_map=None):
    if data:
        assert len(predictions) == len(data)
        
    with open(file, 'w') as fp:
        for i, prediction in enumerate(predictions):
            event_mentions = []
            for event in prediction["events"]:
                arguments = [{"role": a[2], "start": a[0], "end": a[1]} for a in event["arguments"]]
                
                event_mention = {
                    "event_type": event["trigger"][2],  
                    "trigger": {
                        "start": event["trigger"][0], 
                        "end": event["trigger"][1], 
                    }, 
                    "arguments": arguments
                }
                
                if data and offset_map:
                    event_mention["trigger"]["text"] = data[i]["text"][offset_map[i][event_mention["trigger"]["start"]][0]:offset_map[i][event_mention["trigger"]["end"]-1][1]]
                    event_mention["trigger"]["offset_start"] = offset_map[i][event_mention["trigger"]["start"]][0]
                    event_mention["trigger"]["offset_end"] = offset_map[i][event_mention["trigger"]["end"]-1][1]
                    for a in arguments:
                        a["text"] = data[i]["text"][offset_map[i][a["start"]][0]:offset_map[i][a["end"]-1][1]]
                        a["offset_start"] = offset_map[i][a["start"]][0]
                        a["offset_end"] = offset_map[i][a["end"]-1][1]
                else:
                    event_mention["trigger"]["text"] = " ".join(prediction["tokens"][event_mention["trigger"]["start"]:event_mention["trigger"]["end"]])
                    for a in arguments:
                        a["text"] = " ".join(prediction["tokens"][a["start"]:a["end"]])
                
                event_mentions.append(event_mention)
                
            out = {
                "doc_id": prediction["doc_id"], 
                "wnd_id": prediction["wnd_id"], 
                "tokens": prediction["tokens"], 
                "event_mentions": event_mentions, 
            }
                
            if data:
                out["text"] = data[i]["text"]
                
            fp.write(json.dumps(out)+"\n")
