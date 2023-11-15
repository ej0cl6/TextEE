from collections import defaultdict


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def convert_arguments(roles, trigger_type_stoi):
    args = set()
    for r in roles:
        role = r[2]
        arg_start = r[1][0]
        arg_end = r[1][1]
        trigger_label = trigger_type_stoi[r[0][2]]
        args.add((arg_start, arg_end, trigger_label, role))
    return args


def score_graphs(gold_graphs, pred_graphs, trigger_type_stoi, 
                 relation_directional=False):
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0

    for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):
        # Entity
        gold_entities = [(s, e, l) for s, e, l, _ in gold_graph.entities]
        pred_entities = [(s, e, l) for s, e, l, _ in pred_graph.entities]
        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities
                              if entity in gold_entities])

        # Relation
        gold_relations = gold_graph.relations
        pred_relations = pred_graph.relations
        gold_rel_num += len(gold_relations)
        pred_rel_num += len(pred_relations)
        for arg1, arg2, rel_type, _ in pred_relations:
            arg1_start, arg1_end, _ = arg1
            arg2_start, arg2_end, _ = arg2
            for arg1_gold, arg2_gold, rel_type_gold, _ in gold_relations:
                arg1_start_gold, arg1_end_gold, _ = arg1_gold
                arg2_start_gold, arg2_end_gold, _ = arg2_gold
                if relation_directional:
                    if (arg1_start == arg1_start_gold and
                        arg1_end == arg1_end_gold and
                        arg2_start == arg2_start_gold and
                        arg2_end == arg2_end_gold
                    ) and rel_type == rel_type_gold:
                        rel_match_num += 1
                        break
                else:
                    if ((arg1_start == arg1_start_gold and
                            arg1_end == arg1_end_gold and
                            arg2_start == arg2_start_gold and
                            arg2_end == arg2_end_gold) or (
                        arg1_start == arg2_start_gold and
                        arg1_end == arg2_end_gold and
                        arg2_start == arg1_start_gold and
                        arg2_end == arg1_end_gold
                    )) and rel_type == rel_type_gold:
                        rel_match_num += 1
                        break

        # Trigger
        gold_triggers = [(s, e, l) for s, e, l, _ in gold_graph.triggers]
        pred_triggers = [(s, e, l) for s, e, l, _ in pred_graph.triggers]
        gold_trigger_num += len(gold_triggers)
        pred_trigger_num += len(pred_triggers)
        for trg_start, trg_end, event_type in pred_triggers:
            matched = [item for item in gold_triggers
                       if item[0] == trg_start and item[1] == trg_end]
            if matched:
                trigger_idn_num += 1
                if matched[0][-1] == event_type:
                    trigger_class_num += 1

        # Argument
        gold_args = convert_arguments(gold_graph.roles, trigger_type_stoi)
        pred_args = convert_arguments(pred_graph.roles, trigger_type_stoi)
        gold_arg_num += len(gold_args)
        pred_arg_num += len(pred_args)
        for pred_arg in pred_args:
            arg_start, arg_end, event_type, role = pred_arg
            gold_idn = {item for item in gold_args
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_class_num += 1

    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_idn_num)
    trigger_prec, trigger_rec, trigger_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_class_num)
    relation_prec, relation_rec, relation_f = compute_f1(
        pred_rel_num, gold_rel_num, rel_match_num)
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print("---------------------------------------------------------------------")
    print('Entity     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        entity_prec * 100.0, ent_match_num, pred_ent_num, 
        entity_rec * 100.0, ent_match_num, gold_ent_num, entity_f * 100.0))
    
    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        trigger_id_prec * 100.0, trigger_idn_num, pred_trigger_num, 
        trigger_id_rec * 100.0, trigger_idn_num, gold_trigger_num, trigger_id_f * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        trigger_prec * 100.0, trigger_class_num, pred_trigger_num,
        trigger_rec * 100.0, trigger_class_num, gold_trigger_num, trigger_f * 100.0))
    
    print("---------------------------------------------------------------------")
    print('Relation   - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        relation_prec * 100.0, rel_match_num, pred_rel_num, 
        relation_rec * 100.0, rel_match_num, gold_arg_num, relation_f * 100.0))
    
    print("---------------------------------------------------------------------")
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        role_id_prec * 100.0, arg_idn_num, pred_arg_num, 
        role_id_rec * 100.0, arg_idn_num, gold_arg_num, role_id_f * 100.0))
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        role_prec * 100.0, arg_class_num, pred_arg_num, 
        role_rec * 100.0, arg_class_num, gold_arg_num, role_f * 100.0))
    
    print("---------------------------------------------------------------------")
    scores = {
        'entity': {'prec': entity_prec, 'rec': entity_rec, 'f': entity_f},
        'trigger': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
        'trigger_id': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
                       'f': trigger_id_f},
        'role': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
        'role_id': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f},
        'relation': {'prec': relation_prec, 'rec': relation_rec,
                     'f': relation_f}
    }
    return scores
