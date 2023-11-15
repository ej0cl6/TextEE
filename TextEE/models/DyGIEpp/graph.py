import torch
import time
import ipdb

def del_list_inplace(l, id_to_del):
    for i in sorted(id_to_del, reverse=True):
        del(l[i])


class Graph(object):
    def __init__(self, entities, triggers, relations, roles, vocabs, gold=True):
        """
        :param entities (list): A list of entities represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param triggers (list): A list of triggers represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param relations (list): A list of relations represented as a tuple of
        ((start_offset,end_offset,ent_type), (start_offset,end_offset,ent_type), label_idx). 
        :param roles: A list of roles represented as a tuple of 
        ((start_offset,end_offset,trigger_type), (start_offset,end_offset,ent_type), label_idx).
        :param vocabs (dict): Label type vocabularies.
        :param gold (bool): A marker that mark the graph is a gold annotation or not.
        """
        self.entities = [(s,e,l,gold) for (s,e,l) in entities] if gold is not None else entities
        # we write into this form because of "copy" function
        self.triggers = [(s,e,l,gold) for (s,e,l) in triggers] if gold is not None else triggers
        self.relations = [(e1,e2,l,gold) for (e1,e2,l) in relations] if gold is not None else relations
        self.roles = [(t1,e2,l,gold) for (t1,e2,l) in roles] if gold is not None else roles
        
        self.vocabs = vocabs
        self.entity_type_itos = vocabs.get('entity_type_itos', None)
        self.event_type_itos = vocabs.get('event_type_itos', None)
        self.relation_type_itos = vocabs.get('relation_type_itos', None)
        self.role_type_itos = vocabs.get('role_type_itos', None)

        self.entity_num = len(entities)
        self.trigger_num = len(triggers)
        self.relation_num = len(relations)
        self.role_num = len(roles)

        # subscores
        self.entity_scores = [0]*self.entity_num
        self.trigger_scores = [0]*self.trigger_num
        self.relation_scores = [0]*self.relation_num
        self.role_scores = [0]*self.role_num

        # node span embedding
        self.entity_emb = [None]*self.entity_num
        self.trigger_emb = [None]*self.trigger_num

        # span map -- this is for checking overlapping
        self.entity_map = {}
        self.trigger_map = {}
        self.relation_map = {}
        self.role_map = {}
        for idx,ent in enumerate(self.entities):
            if (ent[0], ent[1]) in self.entity_map.keys():
                print('entity span duplication in initialization')
            else:
                self.entity_map[(ent[0], ent[1])] = idx
        for idx,tri in enumerate(self.triggers):
            if (tri[0], tri[1]) in self.trigger_map.keys():
                #raise ValueError('trigger span duplication in initialization')
                print('trigger span duplication in initialization')
                # ipdb.set_trace()
            else:
                self.trigger_map[(tri[0], tri[1])] = idx
        
        for idx,rel in enumerate(self.relations):
            if (rel[0][0], rel[0][1], rel[1][0], rel[1][1]) in self.relation_map.keys():
                raise ValueError('relation span duplication in initialization')
            else:
                self.relation_map[(rel[0][0], rel[0][1], rel[1][0], rel[1][1])] = idx
        
        for idx,role in enumerate(self.roles):
            if (role[0][0], role[0][1], role[1][0], role[1][1]) in self.role_map.keys():
                #raise ValueError('role span duplication in initialization')
                print('role span duplication in initialization')
                #pass
            else:
                self.role_map[(role[0][0], role[0][1], role[1][0], role[1][1])] = idx

    def __eq__(self, other):
        # TODO (I-Hung) Haven't decide whether Gold marker should also be consider
        if isinstance(other, Graph):
            equal = (self.entities == other.entities and
                     self.triggers == other.triggers and
                     self.relations == other.relations and
                     self.roles == other.roles )
            return equal
        return False

    def to_dict(self):
        """Convert a graph to a dict object
        :return (dict): A dictionary representing the graph, where label indices
        have been replaced with label strings.
        """
        entities = [[i, j, self.entity_type_itos[k]] for i, j, k, _ in self.entities]
        triggers = [[i, j, self.event_type_itos[k]] for i, j, k, _ in self.triggers]
        relations = []
        for rel in self.relations:
            arg1_span = (rel[0][0], rel[0][1])
            arg1 = self.entity_map[arg1_span]
            arg2_span = (rel[1][0], rel[1][1])
            arg2 = self.entity_map[arg2_span]
            relations.append([arg1, arg2, self.relation_type_itos[rel[2]]])
        roles = []
        for role in self.roles:
            tri_span = (role[0][0], role[0][1])
            tri = self.trigger_map[tri_span]
            arg_span = (role[1][0], role[1][1])
            arg = self.entity_map[arg_span]
            roles.append([tri, arg, self.role_type_itos[role[2]]])
        return {
            'entities': entities,
            'triggers': triggers,
            'relations': relations,
            'roles': roles,
        }
    def clean_relation(self):
        self.relations = []
        self.relation_num = 0
        self.relation_scores = []
        self.relation_map = {}

    def clean_trigger(self):
        self.triggers = []
        self.trigger_num = 0
        self.trigger_scores = []
        self.trigger_emb = []
        self.trigger_map = {}

    def clean_entity(self):
        self.entities = []
        self.entity_num = 0
        self.entity_scores = []
        self.entity_emb = []
        self.entity_map = {}

    def clean_role(self):
        self.roles = []
        self.role_num = 0
        self.role_scores = []
        self.role_map = {}

    def copy(self):
        """Make a copy of the graph
        :return (Graph): a copy of the current graph.
        """
        graph = Graph(
            entities=self.entities.copy(),
            triggers=self.triggers.copy(),
            relations=self.relations.copy(),
            roles=self.roles.copy(),
            vocabs=self.vocabs,
            gold=None
        )
        graph.entity_scores = self.entity_scores.copy()
        graph.trigger_scores = self.trigger_scores.copy()
        graph.trigger_emb = self.trigger_emb.copy()
        graph.entity_emb = self.entity_emb.copy()
        graph.relation_scores = self.relation_scores.copy()
        graph.role_scores = self.role_scores.copy()
        return graph

    def clean(self, clean_emb=True, relation_directional=True, symmetric_relations=None):
        '''
        This function is used for cleaning entities/triggers/relations/roles
        that have labels 0 (We assume label 0 is 'O')
        '''
        assert symmetric_relations is None
        # clean entities
        remove_entity_idx = []
        for idx, (s,e,k,_) in enumerate(self.entities):
            if k == 0:
                remove_entity_idx.append(idx)
        del_list_inplace(self.entities, remove_entity_idx)
        del_list_inplace(self.entity_scores, remove_entity_idx)
        if clean_emb:
            self.entity_emb = [None] * len(self.entities)
        self.entity_map = {}
        for idx,ent in enumerate(self.entities):
            if (ent[0], ent[1]) in self.entity_map.keys():
                print('entity span duplication in clean')
            else:
                self.entity_map[(ent[0], ent[1])] = idx

        # clean triggers
        remove_trigger_idx = []
        for idx, (s,e,k,_) in enumerate(self.triggers):
            if k == 0:
                remove_trigger_idx.append(idx)
        del_list_inplace(self.triggers, remove_trigger_idx)
        del_list_inplace(self.trigger_scores, remove_trigger_idx)
        if clean_emb:
            self.trigger_emb = [None] * len(self.triggers)
        self.trigger_map = {}
        for idx,tri in enumerate(self.triggers):
            if (tri[0], tri[1]) in self.trigger_map.keys():
                # raise ValueError('trigger span duplication in clean')
                pass
            else:
                self.trigger_map[(tri[0], tri[1])] = idx

        # clean relations
        relations = [[i, j, k, l] for (i, j, k, l) in self.relations]
        remove_relation_idx = []
        for idx, (e1, e2, k, _) in enumerate(relations):
            if (e1[0], e1[1]) not in self.entity_map.keys():
                remove_relation_idx.append(idx)
                continue
            if (e2[0], e2[1]) not in self.entity_map.keys():
                remove_relation_idx.append(idx)
                continue
            if k == 0:
                remove_relation_idx.append(idx)
                continue
        del_list_inplace(relations, remove_relation_idx)
        del_list_inplace(self.relation_scores, remove_relation_idx)
        self.relations = [tuple(r) for r in relations]
        relations = [(i, j, k, g, l) for (i, j, k, g), l in zip(self.relations, self.relation_scores)]
        if not relation_directional:
            # rebuild relation map
            self.relation_map = {}
            for idx,rel in enumerate(relations):
                if (rel[0][0], rel[0][1], rel[1][0], rel[1][1]) in self.relation_map.keys():
                    raise ValueError('relation span duplication in clean')
                else:
                    self.relation_map[(rel[0][0], rel[0][1], rel[1][0], rel[1][1])] = idx

            relations_tmp = []
            for i, j, k, g, l in relations:
                if (j[0], j[1], i[0], i[1]) in self.relation_map.keys():
                    if i[0] <= j[0]: # follow the smaller one's prediction
                        relations_tmp.append((i, j, k, g, l))
                        relations_tmp.append((j, i, k, g, l))
                else:
                    relations_tmp.append((i, j, k, g, l))
                    relations_tmp.append((j, i, k, g, l))
            relations = relations_tmp
        self.relations = [(i, j, k, g) for i, j, k, g,_ in relations]
        self.relation_scores = [l for _, _, _, _, l in relations]
        # rebuild relation map
        self.relation_map = {}
        for idx,rel in enumerate(self.relations):
            if (rel[0][0], rel[0][1], rel[1][0], rel[1][1]) in self.relation_map.keys():
                raise ValueError('relation span duplication in clean')
            else:
                self.relation_map[(rel[0][0], rel[0][1], rel[1][0], rel[1][1])] = idx
        
        # clean roles
        roles = [[i, j, k, g]  for (i, j, k, g) in self.roles]
        remove_role_idx = []
        for idx, (t, e, k, g) in enumerate(roles):
            if (t[0], t[1]) not in self.trigger_map.keys():
                remove_role_idx.append(idx)
                continue
            if (e[0], e[1]) not in self.entity_map.keys():
                remove_role_idx.append(idx)
                continue
            if k == 0:
                remove_role_idx.append(idx)
                continue
        del_list_inplace(roles, remove_role_idx)
        del_list_inplace(self.role_scores, remove_role_idx)       
        self.roles = [tuple(r) for r in roles]
        # rebuild role map
        self.role_map = {}
        for idx,role in enumerate(self.roles):
            if (role[0][0], role[0][1], role[1][0], role[1][1]) in self.role_map.keys():
                # raise ValueError('role span duplication in clean')
                pass
            else:
                self.role_map[(role[0][0], role[0][1], role[1][0], role[1][1])] = idx

        self.entity_num = len(self.entities)
        self.trigger_num = len(self.triggers)
        self.relation_num = len(self.relations)
        self.role_num = len(self.roles)
 
    def add_entity(self, start, end, label, emb=None, score_norm=0, gold=False):
        """Add an entity mention to the graph.
        :param start (int): Start token offset of the entity mention.
        :param end (int): End token offset of the entity mention + 1.
        :param label (int): Index of the entity type label.
        """
        # check whether this entity is duplicate
        if (start, end) not in self.entity_map.keys():
            self.entity_map[(start, end)]=self.entity_num
            self.entities.append((start, end, label, gold))
            self.entity_num = len(self.entities)
            self.entity_scores.append(score_norm)
            self.entity_emb.append(emb)
            return True
        else:
            #print('Duplicate entity for span ({}, {})'.format(start, end))
            return False

    def add_trigger(self, start, end, label, emb=None, score_norm=0, gold=False):
        """Add an event trigger to the graph.
        :param start (int): Start token offset of the trigger.
        :param end (int): End token offset of the trigger + 1.
        :param label (int): Index of the event type label.
        :param score (float): Label score.
        :param gold (bool): Marker that mark this trigger is gold or not
        """
        # check whether this trigger is duplicate
        if (start, end) not in self.trigger_map.keys():
            self.trigger_map[(start, end)]=self.trigger_num
            self.triggers.append((start, end, label, gold))
            self.trigger_num = len(self.triggers)
            self.trigger_scores.append(score_norm)
            self.trigger_emb.append(emb)
            return True
        else:
            #print('Duplicate trigger for span ({}, {})'.format(start, end))
            return False

    def add_relation(self, ent_1, ent_2, label, score_norm=0, gold=False):
        """Add a relation edge to the graph.
        :param ent_1 (tuple(int, int, str)): start& end of the entity node 1.
        :param ent_2 (tuple(int, int, str)): start& end of the entity node 2.
        :param label (int): Index of the relation type label.
        :param score (float): Label score.
        :param gold (bool): Marker that mark this relation is gold or not
        """
        assert ((ent_1[0], ent_1[1]) in self.entity_map.keys())
        assert ((ent_2[0], ent_2[1]) in self.entity_map.keys())
        if (ent_1[0], ent_1[1], ent_2[0], ent_2[1]) not in self.relation_map.keys():
            self.relation_map[(ent_1[0], ent_1[1], ent_2[0], ent_2[1])]=self.relation_num
            self.relations.append((ent_1, ent_2, label, gold))
            self.relation_num = len(self.relations)
            self.relation_scores.append(score_norm)
            return True
        else:
            return False

    def add_role(self, tri, ent, label, score_norm=0, gold=False):
        """Add an event-argument link edge to the graph.
        :param tri (tuple(int, int, str)): start& end of the trigger node.
        :param ent (tuple(int, int, str)): start& end of the entity node.
        :param label (int): Index of the role label.
        :param score (float): Label score.
        :param gold (bool): Marker that mark this role is gold or not
        """
        assert ((ent[0], ent[1]) in self.entity_map.keys())
        assert ((tri[0], tri[1]) in self.trigger_map.keys())
        if (tri[0], tri[1], ent[0], ent[1]) not in self.role_map.keys():
            self.role_map[(tri[0], tri[1], ent[0], ent[1])]=self.role_num
            self.roles.append((tri, ent, label, gold))
            self.role_num = len(self.roles)
            self.role_scores.append(score_norm)
            return True
        else:
            return False

    @staticmethod
    def empty_graph(vocabs):
        """Create a graph without any node and edge.
        :param vocabs (dict): Vocabulary object.
        """
        return Graph([], [], [], [], vocabs)


    def clean_non_gold(self, relation_directional=True, symmetric_relations=None):
        '''
        This function is used for cleaning entities/triggers/relations/roles
        that have gold label==False
        '''
        assert symmetric_relations is None
        # clean entities
        remove_entity_idx = []
        for idx, (s,e,_,gold) in enumerate(self.entities):
            if not gold:
                remove_entity_idx.append(idx)
        del_list_inplace(self.entities, remove_entity_idx)
        del_list_inplace(self.entity_scores, remove_entity_idx)
        del_list_inplace(self.entity_emb, remove_entity_idx)
        self.entity_map = {}
        for idx,ent in enumerate(self.entities):
            if (ent[0], ent[1]) in self.entity_map.keys():
                print('entity span duplication in clean')
            else:
                self.entity_map[(ent[0], ent[1])] = idx

        # clean triggers
        remove_trigger_idx = []
        for idx, (s,e,_,gold) in enumerate(self.triggers):
            if not gold:
                remove_trigger_idx.append(idx)
        del_list_inplace(self.triggers, remove_trigger_idx)
        del_list_inplace(self.trigger_scores, remove_trigger_idx)
        del_list_inplace(self.trigger_emb, remove_trigger_idx)
        self.trigger_map = {}
        for idx,tri in enumerate(self.triggers):
            if (tri[0], tri[1]) in self.trigger_map.keys():
                raise ValueError('trigger span duplication in clean')
            else:
                self.trigger_map[(tri[0], tri[1])] = idx
        
        # clean relations
        relations = [[i, j, k, l] for (i, j, k, l) in self.relations]
        remove_relation_idx = []
        for idx, (e1, e2, _, gold) in enumerate(relations):
            if (e1[0], e1[1]) not in self.entity_map.keys():
                remove_relation_idx.append(idx)
                continue
            if (e2[0], e2[1]) not in self.entity_map.keys():
                remove_relation_idx.append(idx)
                continue
            if not gold:
                remove_relation_idx.append(idx)
                continue
        del_list_inplace(relations, remove_relation_idx)
        del_list_inplace(self.relation_scores, remove_relation_idx)
        self.relations = [tuple(r) for r in relations]
        relations = [(i, j, k, g, l) for (i, j, k, g), l in zip(self.relations, self.relation_scores)]
        if not relation_directional:
            # rebuild relation map
            self.relation_map = {}
            for idx,rel in enumerate(relations):
                if (rel[0][0], rel[0][1], rel[1][0], rel[1][1]) in self.relation_map.keys():
                    raise ValueError('relation span duplication in clean')
                else:
                    self.relation_map[(rel[0][0], rel[0][1], rel[1][0], rel[1][1])] = idx

            relations_tmp = []
            for i, j, k, g, l in relations:
                if (j[0], j[1], i[0], i[1]) in self.relation_map.keys():
                    if i[0] <= j[0]: # follow the smaller one's prediction
                        relations_tmp.append((i, j, k, g, l))
                        relations_tmp.append((j, i, k, g, l))
                else:
                    relations_tmp.append((i, j, k, g, l))
                    relations_tmp.append((j, i, k, g, l))
            relations = relations_tmp
        self.relations = [(i, j, k, g) for i, j, k, g,_ in relations]
        self.relation_scores = [l for _, _, _, _, l in relations]
        # rebuild relation map
        self.relation_map = {}
        for idx,rel in enumerate(self.relations):
            if (rel[0][0], rel[0][1], rel[1][0], rel[1][1]) in self.relation_map.keys():
                raise ValueError('relation span duplication in initialization')
            else:
                self.relation_map[(rel[0][0], rel[0][1], rel[1][0], rel[1][1])] = idx
        
        # clean roles
        roles = [[i, j, k, g]  for (i, j, k, g) in self.roles]
        remove_role_idx = []
        for idx, (t, e, _, gold) in enumerate(roles):
            if (t[0], t[1]) not in self.trigger_map.keys():
                remove_role_idx.append(idx)
                continue
            if (e[0], e[1]) not in self.entity_map.keys():
                remove_role_idx.append(idx)
                continue
            if not gold:
                remove_role_idx.append(idx)
                continue
        del_list_inplace(roles, remove_role_idx)
        del_list_inplace(self.role_scores, remove_role_idx)       
        self.roles = [tuple(r) for r in roles]
        # rebuild role map
        self.role_map = {}
        for idx,role in enumerate(self.roles):
            if (role[0][0], role[0][1], role[1][0], role[1][1]) in self.role_map.keys():
                raise ValueError('role span duplication in initialization')
            else:
                self.role_map[(role[0][0], role[0][1], role[1][0], role[1][1])] = idx
        self.entity_num = len(self.entities)
        self.trigger_num = len(self.triggers)
        self.relation_num = len(self.relations)
        self.role_num = len(self.roles)
