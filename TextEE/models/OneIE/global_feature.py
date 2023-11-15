import itertools

import numpy as np

from collections import Counter

import ipdb


def generate_global_feature_maps(vocabs, valid_patterns):
    """
    Note that feature maps here refer to "feature-index mappings", not feature
    maps in CNNs.
    :param vocabs: vocabularies.
    :param valid_patterns: valid patterns (only event-role patterns are used).
    :return (dict): a dictionary of feature-index maps.
    """
    event_type_vocab = vocabs['event_type']
    entity_type_vocab = vocabs['entity_type']
    role_type_vocab = vocabs['role_type']
    relation_type_vocab = vocabs['relation_type']
    event_role = valid_patterns['event_role']

    # 1. role role: the number of entities that act as <role_i> and <role_j>
    # arguments at the same time
    role_role_map = set()
    for role1 in role_type_vocab.values():
        for role2 in role_type_vocab.values():
            if role1 and role2:
                if role1 < role2:
                    key = role1 * 1000 + role2
                else:
                    key = role2 * 1000 + role1
                role_role_map.add(key)
    role_role_map = sorted(list(role_role_map))
    role_role_map = {k: i for i, k in enumerate(role_role_map)}

    # 2. event role num: the number of <event_type_i> events with <number>
    # <role_j> arguments
    event_role_num_map = list()
    for event in event_type_vocab.values():
        for role in role_type_vocab.values():
            if event and role:
                key = event * 100000 + role * 100
                event_role_num_map.append(key + 1)
                event_role_num_map.append(key + 2)
    event_role_num_map.sort()
    event_role_num_map = {k: i for i, k in enumerate(event_role_num_map)}

    # 3. role entity: the number of occurrences of <entity_type_i> and <role_j>
    # combination
    role_entity_map = list()
    for role in role_type_vocab.values():
        for entity in entity_type_vocab.values():
            if role and entity:
                role_entity_map.append(role * 1000 + entity)
    role_entity_map.sort()
    role_entity_map = {k: i for i, k in enumerate(role_entity_map)}

    # 4. multiple role
    multi_role_map = [role for role in role_type_vocab.values() if role]
    multi_role_map.sort()
    multi_role_map = {k: i for i, k in enumerate(multi_role_map)}

    # 5. event role event role: the number of entities that act as a <role_i>
    # argument of an <event_type_j> event and a <role_k> argument of an
    # <event_type_l> event at the same time
    event_role_event_role_map = set()
    for event_role1 in event_role:
        for event_role2 in event_role:
            event1 = event_role1 // 1000
            event2 = event_role2 // 1000
            role1 = event_role1 % 1000
            role2 = event_role2 % 1000
            if event1 < event2:
                key = event1 * 1000000000 + role1 * 1000000 + event2 * 1000 + role2
            else:
                key = event2 * 1000000000 + role2 * 1000000 + event1 * 1000 + role1
            event_role_event_role_map.add(key)
    event_role_event_role_map = sorted(list(event_role_event_role_map))
    event_role_event_role_map = {k: i for i, k in enumerate(event_role_event_role_map)}

    # 6. relation entity entity: the number of occurrences of <entity_type_i>,
    # <entity_type_j>, and <relation_type_k> combination
    relation_entity_entity_map = set()
    for relation in relation_type_vocab.values():
        for entity1 in entity_type_vocab.values():
            for entity2 in entity_type_vocab.values():
                if relation and entity1 and entity2:
                    key = relation * 1000000
                    if entity1 < entity2:
                        key += entity1 * 1000 + entity2
                    else:
                        key += entity2 * 1000 + entity1
                    relation_entity_entity_map.add(key)
    relation_entity_entity_map = sorted(list(relation_entity_entity_map))
    relation_entity_entity_map = {k: i for i, k in enumerate(relation_entity_entity_map)}

    # 7. relation entity: the number of occurrences of <entity_type_i> and
    # <relation_type_j> combination
    relation_entity_map = [relation * 1000 + entity
                           for relation in relation_type_vocab.values()
                           for entity in entity_type_vocab.values()
                           if relation and entity]
    relation_entity_map.sort()
    relation_entity_map = {k: i for i, k in enumerate(relation_entity_map)}

    # 8. relation role role: the number of occurrences of a <relation_type_i>
    # relation between a <role_j> argument and a <role_k> argument of the same
    # event
    relation_role_role_map = set()
    for relation in relation_type_vocab.values():
        for role1 in role_type_vocab.values():
            for role2 in role_type_vocab.values():
                if relation and role1 and role2:
                    key = relation * 1000000
                    if role1 < role2:
                        key += role1 * 1000 + role2
                    else:
                        key += role2 * 1000 + role1
                    relation_role_role_map.add(key)
    relation_role_role_map = sorted(list(relation_role_role_map))
    relation_role_role_map = {k: i for i, k in enumerate(relation_role_role_map)}

    # 9. multiple relation: the number of entities that have a <relation_type_i>
    # relation with multiple entities
    multi_relation_map = [relation for relation in relation_type_vocab.values()
                          if relation]
    multi_relation_map.sort()
    multi_relation_map = {k: i for i, k in enumerate((multi_relation_map))}

    # 10. relation relation: the number of entities involving in <relation_type_i>
    # and <relation_type_j> relations simultaneously
    relation_relation_map = set()
    for relation1 in relation_type_vocab.values():
        for relation2 in relation_type_vocab.values():
            if relation1 and relation2:
                key = relation1 * 1000 + relation2 if relation1 < relation2 \
                    else relation2 * 1000 + relation1
                relation_relation_map.add(key)
    relation_relation_map = sorted(list(relation_relation_map))
    relation_relation_map = {k: i for i, k in enumerate(relation_relation_map)}

    # 11. multiple event: whether a graph contains more than one <event_type_i>
    # event
    multi_event_map = [event for event in event_type_vocab.values() if event]
    multi_event_map.sort()
    multi_event_map = {k: i for i, k in enumerate(multi_event_map)}

    return {
        'role_role': role_role_map,
        'event_role_num': event_role_num_map,
        'role_entity': role_entity_map,
        'multi_role': multi_role_map,
        'event_role_event_role': event_role_event_role_map,
        'relation_entity_entity': relation_entity_entity_map,
        'relation_entity': relation_entity_map,
        'relation_role_role': relation_role_role_map,
        'multi_relation': multi_relation_map,
        'relation_relation': relation_relation_map,
        'multi_event': multi_event_map
    }


def generate_global_feature_vector(graph,
                                   global_feature_maps,
                                   features=None):
    role_role_map = global_feature_maps['role_role']
    role_role_vec = np.zeros(len((role_role_map)))
    role_entity_map = global_feature_maps['role_entity']
    role_entity_vec = np.zeros(len(role_entity_map))
    event_role_num_map = global_feature_maps['event_role_num']
    event_role_num_vec = np.zeros(len(event_role_num_map))
    multi_role_map = global_feature_maps['multi_role']
    multi_role_vec = np.zeros(len(multi_role_map))
    event_role_event_role_map = global_feature_maps['event_role_event_role']
    event_role_event_role_vec = np.zeros(len(event_role_event_role_map))
    relation_entity_entity_map = global_feature_maps['relation_entity_entity']
    relation_entity_entity_vec = np.zeros(len(relation_entity_entity_map))
    relation_entity_map = global_feature_maps['relation_entity']
    relation_entity_vec = np.zeros(len(relation_entity_map))
    relation_role_role_map = global_feature_maps['relation_role_role']
    relation_role_role_vec = np.zeros(len(relation_role_role_map))
    multi_relation_map = global_feature_maps['multi_relation']
    multi_relation_vec = np.zeros(len(multi_relation_map))
    relation_relation_map = global_feature_maps['relation_relation']
    relation_relation_vec = np.zeros(len(relation_relation_map))
    multi_event_map = global_feature_maps['multi_event']
    multi_event_vec = np.zeros(len(multi_event_map))

    # event argument role related features
    entity_roles = [[] for _ in range(graph.entity_num)]
    entity_event_role = [[] for _ in range(graph.entity_num)]
    event_role_count = [Counter() for _ in range(graph.trigger_num)]
    for trigger_idx, entity_idx, role in graph.roles:
        entity_roles[entity_idx].append(role)
        entity_event_role[entity_idx].append(
            (graph.triggers[trigger_idx][-1], role))
        event_role_count[trigger_idx][role] += 1
        # 3. role entity
        role_entity = role * 1000 + graph.entities[entity_idx][-1]
        if role_entity in role_entity_map:
            role_entity_vec[role_entity_map[role_entity]] += 1
    # 1. role role
    for roles in entity_roles:
        for role1, role2 in itertools.combinations(roles, 2):
            key = role1 * 1000 + role2 if role1 < role2 \
                else role2 * 1000 + role1
            if key in role_role_map:
                role_role_vec[role_role_map[key]] += 1
    # 2. event role num & 4. multiple role
    for event, role_count in enumerate(event_role_count):
        for role, count in role_count.items():
            # to reduce the number of features, we treat numbers > 2 as 2
            key = graph.triggers[event][-1] * 100000 + role * 100 + min(count, 2)
            if key in event_role_num_map:
                event_role_num_vec[event_role_num_map[key]] += 1
            if count > 1 and role in multi_role_map:
                multi_role_vec[multi_role_map[role]] += 1
    # 5. event role event role
    for event_role_pairs in entity_event_role:
        for (event1, role1), (event2, role2) in itertools.combinations(
                event_role_pairs, 2):
            if event1 < event2:
                key = event1 * 1000000000 + role1 * 1000000 + event2 * 1000 + role2
            else:
                key = event2 * 1000000000 + role2 * 1000000 + event1 * 1000 + role1
            if key in event_role_event_role_map:
                event_role_event_role_vec[event_role_event_role_map[key]] += 1

    # relation related features
    entity_role_unique = [set(x) for x in entity_roles]
    entity_relation_count = [Counter() for _ in range(graph.entity_num)]
    for entity_idx1, entity_idx2, relation in graph.relations:
        entity_relation_count[entity_idx1][relation] += 1
        entity_relation_count[entity_idx2][relation] += 1
        entity1 = graph.entities[entity_idx1][-1]
        entity2 = graph.entities[entity_idx2][-1]
        # 6. relation entity entity
        if entity1 < entity2:
            key = relation * 1000000 + entity1 * 1000 + entity2
        else:
            key = relation * 1000000 + entity2 * 1000 + entity1
        if key in relation_entity_entity_map:
            relation_entity_entity_vec[relation_entity_entity_map[key]] += 1
        # 7. relation entity
        key1 = relation * 1000 + entity1
        key2 = relation * 1000 + entity2
        if key1 in relation_entity_map:
            relation_entity_vec[relation_entity_map[key1]] += 1
        if key2 in relation_entity_map:
            relation_entity_vec[relation_entity_map[key2]] += 1
        # 8. relation role role
        roles1 = entity_role_unique[entity_idx1]
        roles2 = entity_role_unique[entity_idx2]
        for role1 in roles1:
            for role2 in roles2:
                if role1 < role2:
                    key = relation * 1000000 + role1 * 1000 + role2
                else:
                    key = relation * 1000000 + role2 * 1000 + role1
                if key in relation_role_role_map:
                    relation_role_role_vec[relation_role_role_map[key]] += 1
    # 9. multiple relation & 10. relation relation
    for relation_count in entity_relation_count:
        relations = []
        for relation, count in relation_count.items():
            relations.append(relation)
            if count > 1:
                relations.append(relation)
                if relation in multi_relation_map:
                    multi_relation_vec[multi_relation_map[relation]] += 1
        for relation1, relation2 in itertools.combinations(relations, 2):
            if relation1 < relation2:
                key = relation1 * 1000 + relation2
            else:
                key = relation2 * 1000 + relation1
            if key in relation_relation_map:
                relation_relation_vec[relation_relation_map[key]] += 1

    # 11. multiple event
    trigger_count = Counter()
    for _, _, trigger in graph.triggers:
        trigger_count[trigger] += 1
    for trigger, count in trigger_count.items():
        if count > 1 and trigger in multi_event_map:
            multi_event_vec[multi_event_map[trigger]] = 1

    feature_vector = np.concatenate(
        [role_role_vec, event_role_num_vec, role_entity_vec,
         multi_role_vec, event_role_event_role_vec, relation_entity_entity_vec,
         relation_entity_vec, relation_role_role_vec,
         multi_relation_vec, relation_relation_vec, multi_event_vec]
    )

    if features:
        vectors = {
            'role_role': role_role_vec,
            'event_role_num': event_role_num_vec,
            'role_entity': role_entity_vec,
            'multi_role': multi_role_vec,
            'event_role_event_role': event_role_event_role_vec,
            'relation_entity_entity': relation_entity_entity_vec,
            'relation_entity': relation_entity_vec,
            'relation_role_role': relation_role_role_vec,
            'multi_relation': multi_relation_vec,
            'relation_relation': relation_relation_vec,
            'multi_event': multi_event_vec
        }
        feature_vector = np.concatenate([vectors[k] for k in features])
    else:
        feature_vector = np.concatenate(
            [role_role_vec, event_role_num_vec, role_entity_vec,
             multi_role_vec, event_role_event_role_vec, relation_entity_entity_vec,
             relation_entity_vec, relation_role_role_vec,
             multi_relation_vec, relation_relation_vec, multi_event_vec]
        )
    return feature_vector
