import ipdb

class BasicTrainer(object):
    def __init__(self, config, type_set=None):
        self.config = config
        self.type_set = type_set
        
    @classmethod
    def add_extra_info_fn(cls, instances, raw_data, config):
        for instance in instances:
            instance["extra_info"] = None
        return instances
        
    def load_model(self, checkpoint=None):
        pass
    
    def train(self, train_data, dev_data, **kwargs):
        pass
    
    def predict(self, data, **kwargs):
        pass