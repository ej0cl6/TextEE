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
    if task == "E2E":
        data, type_set = load_E2E_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "ED":
        data, type_set = load_ED_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif task == "EAE":
        data, type_set = load_EAE_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "EARL":
        data, type_set = load_EARL_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return data, type_set

def load_all_data(config, add_extra_info_fn):
    if config.task == "E2E":
        train_data, train_type_set = load_E2E_data(config.train_file, add_extra_info_fn, config)
        dev_data, dev_type_set = load_E2E_data(config.dev_file, add_extra_info_fn, config)
        test_data, test_type_set = load_E2E_data(config.test_file, add_extra_info_fn, config)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "ED":
        train_data, train_type_set = load_ED_data(config.train_file, add_extra_info_fn, config)
        dev_data, dev_type_set = load_ED_data(config.dev_file, add_extra_info_fn, config)
        test_data, test_type_set = load_ED_data(config.test_file, add_extra_info_fn, config)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"]}
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif config.task == "EAE":
        train_data, train_type_set = load_EAE_data(config.train_file, add_extra_info_fn, config)
        dev_data, dev_type_set = load_EAE_data(config.dev_file, add_extra_info_fn, config)
        test_data, test_type_set = load_EAE_data(config.test_file, add_extra_info_fn, config)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "EARL":
        train_data, train_type_set = load_EARL_data(config.train_file, add_extra_info_fn, config)
        dev_data, dev_type_set = load_EARL_data(config.dev_file, add_extra_info_fn, config)
        test_data, test_type_set = load_EARL_data(config.test_file, add_extra_info_fn, config)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return train_data, dev_data, test_data, type_set

def load_E2E_data(file, add_extra_info_fn, config):

    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    data = [json.loads(line) for line in lines]
    
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
    
    logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
        len(new_instances), len(trigger_type_set), len(role_type_set), file))
    
    return new_instances, type_set

def load_ED_data(file, add_extra_info_fn, config):

    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    data = [json.loads(line) for line in lines]
    
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
    
    logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
        len(new_instances), len(trigger_type_set), file))
    
    return new_instances, type_set

def load_EAE_data(file, add_extra_info_fn, config):

    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    data = [json.loads(line) for line in lines]
    
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
    
    logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
        len(new_instances), len(trigger_type_set), len(role_type_set), file))
    
    return new_instances, type_set

def load_EARL_data(file, add_extra_info_fn, config):

    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    data = [json.loads(line) for line in lines]
    
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
    
    logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
        len(new_instances), len(trigger_type_set), len(role_type_set), file))
    
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
