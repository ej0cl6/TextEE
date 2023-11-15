class Graph(object):
    def __init__(self, entities, triggers, relations, roles, vocabs, mentions=None):
        """
        :param entities (list): A list of entities represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param triggers (list): A list of triggers represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param relations (list): A list of relations represented as a tuple of
        (entity_idx_1, entity_idx_2, label_idx). As we do not consider the
        direction of relations (list), it is better to have entity_idx_1 <
        entity_idx2.
        :param roles: A list of roles represented as a tuple of (trigger_idx_1,
        entity_idx_2, label_idx).
        :param vocabs (dict): Label type vocabularies.
        """
        self.entities = entities
        self.triggers = triggers
        self.relations = relations
        self.roles = roles
        self.vocabs = vocabs
        self.mentions = [] if mentions is None else mentions

        self.entity_num = len(entities)
        self.trigger_num = len(triggers)
        self.relation_num = len(relations)
        self.role_num = len(roles)
        self.graph_local_score = 0.0

        # subscores
        self.entity_scores = []
        self.trigger_scores = []
        self.relation_scores = []
        self.role_scores = []

    def __eq__(self, other):
        if isinstance(other, Graph):
            equal = (self.entities == other.entities and
                     self.triggers == other.triggers and
                     self.relations == other.relations and
                     self.roles == other.roles and
                     self.mentions == other.mentions)
            return equal
        return False


    def to_dict(self):
        """Convert a graph to a dict object
        :return (dict): A dictionary representing the graph, where label indices
        have been replaced with label strings.
        """
        entity_itos = {i: s for s, i in self.vocabs['entity_type'].items()}
        trigger_itos = {i: s for s, i in self.vocabs['event_type'].items()}
        relation_itos = {i: s for s, i in self.vocabs['relation_type'].items()}
        role_itos = {i: s for s, i in self.vocabs['role_type'].items()}
        mention_itos = {i: s for s, i in self.vocabs['mention_type'].items()}

        entities = [[i, j, entity_itos[k], mention_itos[l]] for (i, j, k), (_, _, l) in zip(self.entities, self.mentions)]
        triggers = [[i, j, trigger_itos[k]] for (i, j, k) in self.triggers]
        relations = [[i, j, relation_itos[k]] for (i, j, k) in self.relations]
        roles = [[i, j, role_itos[k]] for (i, j, k) in self.roles]

        return {
            'entities': entities,
            'triggers': triggers,
            'relations': relations,
            'roles': roles,
        }

    def __str__(self):
        return str(self.to_dict())

    def copy(self):
        """Make a copy of the graph
        :return (Graph): a copy of the current graph.
        """
        graph = Graph(
            entities=self.entities.copy(),
            triggers=self.triggers.copy(),
            relations=self.relations.copy(),
            roles=self.roles.copy(),
            mentions=self.mentions.copy(),
            vocabs=self.vocabs
        )
        graph.graph_local_score = self.graph_local_score
        graph.entity_scores = self.entity_scores
        graph.trigger_scores = self.trigger_scores
        graph.relation_scores = self.relation_scores
        graph.role_scores = self.role_scores
        return graph

    def clean(self, relation_directional=False, symmetric_relations=None):

        entities = [(i, j, k, l) for (i, j, k), l in zip(self.entities, self.entity_scores)]
        triggers = [(i, j, k, l) for (i, j, k), l in zip(self.triggers, self.trigger_scores)]
        relations = [(i, j, k, l) for (i, j, k), l in zip(self.relations, self.relation_scores)]
        roles = [(i, j, k, l) for (i, j, k), l in zip(self.roles, self.role_scores)]

        # clean relations
        if relation_directional and symmetric_relations:
            relation_itos = {i: s for s, i in self.vocabs['relation_type'].items()}
            # relations = []
            relations_tmp = []
            # for i, j, k in self.relations:
            for i, j, k, l in relations:
                if relation_itos[k] not in symmetric_relations:
                    # relations.append((i, j, k))
                    relations_tmp.append((i, j, k, l))
                else:
                    if j < i:
                        i, j = j, i
                    relations_tmp.append((i, j, k, l))
            # self.relations = relations
            relations = relations_tmp

        self.entities = [(i, j, k) for i, j, k, _ in entities]
        self.entity_scores = [l for _, _, _, l in entities]
        self.triggers = [(i, j, k) for i, j, k, _ in triggers]
        self.trigger_scores = [l for _, _, _, l in triggers]
        self.relations = [(i, j, k) for i, j, k, _ in relations]
        self.relation_scores = [l for _, _, _, l in relations]
        self.roles = [(i, j, k) for i, j, k, _ in roles]
        self.role_scores = [l for _, _, _, l in roles]

    def add_entity(self, start, end, label, score=0, score_norm=0):
        """Add an entity mention to the graph.
        :param start (int): Start token offset of the entity mention.
        :param end (int): End token offset of the entity mention + 1.
        :param label (int): Index of the entity type label.
        :param score (float): Label score.
        """
        self.entities.append((start, end, label))
        self.entity_num = len(self.entities)
        self.graph_local_score += score
        self.entity_scores.append(score_norm)

    def add_trigger(self, start, end, label, score=0, score_norm=0):
        """Add an event trigger to the graph.
        :param start (int): Start token offset of the trigger.
        :param end (int): End token offset of the trigger + 1.
        :param label (int): Index of the event type label.
        :param score (float): Label score.
        """
        self.triggers.append((start, end, label))
        self.trigger_num = len(self.triggers)
        self.graph_local_score += score
        self.trigger_scores.append(score_norm)

    def add_relation(self, idx1, idx2, label, score=0, score_norm=0):
        """Add a relation edge to the graph.
        :param idx1 (int): Index of the entity node 1.
        :param idx2 (int): Index of the entity node 2.
        :param label (int): Index of the relation type label.
        :param score (float): Label score.
        """
        # assert idx1 < self.entity_num and idx2 < self.entity_num
        if label:
            self.relations.append((idx1, idx2, label))
            self.relation_scores.append(score_norm)
        self.relation_num = len(self.relations)
        self.graph_local_score += score

    def add_role(self, idx1, idx2, label, score=0, score_norm=0):
        """Add an event-argument link edge to the graph.
        :param idx1 (int): Index of the trigger node.
        :param idx2 (int): Index of the entity node.
        :param label (int): Index of the role label.
        :param score (float): Label score.
        """
        # assert idx1 < self.trigger_num and idx2 < self.entity_num
        # self.roles.append((idx1, idx2, label))
        if label:
            self.roles.append((idx1, idx2, label))
            self.role_scores.append(score_norm)
        self.role_num = len(self.roles)
        self.graph_local_score += score

    @staticmethod
    def empty_graph(vocabs):
        """Create a graph without any node and edge.
        :param vocabs (dict): Vocabulary object.
        """
        return Graph([], [], [], [], vocabs)

    def to_label_idxs(self, max_entity_num, max_trigger_num,
                      relation_directional=False,
                      symmetric_relation_idxs=None):
        """Generate label index tensors (which are actually list objects not
        Pytorch tensors) to gather calculated scores.
        :param max_entity_num: Max entity number of the batch.
        :param max_trigger_num: Max trigger number of the batch.
        :return: Index and mask tensors.
        """
        entity_idxs = [i[-1] for i in self.entities] + [0] * (max_entity_num - self.entity_num)
        entity_mask = [1] * self.entity_num + [0] * (max_entity_num - self.entity_num)

        trigger_idxs = [i[-1] for i in self.triggers] + [0] * (max_trigger_num - self.trigger_num)
        trigger_mask = [1] * self.trigger_num + [0] * (max_trigger_num - self.trigger_num)

        relation_idxs = [0] * max_entity_num * max_entity_num
        relation_mask = [1 if i < self.entity_num and j < self.entity_num and i != j else 0
                         for i in range(max_entity_num) for j in range(max_entity_num)]
        for i, j, relation in self.relations:
            relation_idxs[i * max_entity_num + j] = relation
            if not relation_directional:
                relation_idxs[j * max_entity_num + i] = relation
            if relation_directional and symmetric_relation_idxs and relation in symmetric_relation_idxs:
                relation_idxs[j * max_entity_num + i] = relation
        

        role_idxs = [0] * max_trigger_num * max_entity_num
        for i, j, role in self.roles:
            role_idxs[i * max_entity_num + j] = role
        role_mask = [1 if i < self.trigger_num and j < self.entity_num else 0
                     for i in range(max_trigger_num) for j in range(max_entity_num)]

        return (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask,
        )

