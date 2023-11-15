import logging
import numpy as np
import torch
import torch.nn as nn
import ipdb

logger = logging.getLogger(__name__)

class Projector(nn.Module):
    def __init__(self, config, model_config, projector_type):
        super().__init__()
        self.config = config
        self.model_config = model_config
        if projector_type=='AttIndep':
            self.projector = AttIndepProjector(config, model_config)
        elif projector_type=='LinearIndep':
            self.projector = LinearIndepProjector(config, model_config)
        else:
            raise ValueError("Model type {} does not support yet.".format(self.config.model_type))

    def project(self, source_embedding=None, embed_query=None):
        """from embedding to prefix weights
        source_embedding: the embedding from rich representation, for example optimus, amr encoder
        embed_query: the embedding that served as query embedding, usually generated from range_function 
        """
        if source_embedding is not None:
            batch_size = source_embedding[0].size(0)
        elif embed_query is not None:
            batch_size = embed_query.size(0)
        else:
            raise ValueError("At least one of source embedding or embedding query should be not None.")
        flat_prefix = self.projector(source_embedding, embed_query)
        prefix_key, prefix_value = flat_prefix.chunk(2, -1)
        prefix_key = prefix_key.view(batch_size, self.model_config.num_hidden_layers, self.config.prefix_length, self.model_config.encoder_attention_heads, -1).contiguous()
        prefix_value = prefix_value.view(batch_size, self.model_config.num_hidden_layers, self.config.prefix_length, self.model_config.encoder_attention_heads, -1).contiguous()
        return (prefix_key, prefix_value)

    def save(self, save_path):
        torch.save(self.projector.state_dict(), save_path)

    def load(self, load_path):
        self.projector.load_state_dict(torch.load(load_path, map_location=f'cuda:{self.config.gpu_device}'))

    def freeze(self):
        for param in self.projector.parameters():
            param.requires_grad=False

class AttIndepProjector(nn.Module):
    """
    This is projector use attention mechanism to calculate EACH embed_query's mapping to source embedding
    """
    def __init__(self, config, model_config):
        super().__init__()
        self.config = config
        self.model_config = model_config
        logger.info(f'Using projector {self.__class__.__name__}')

        self.projector = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.model_config.d_model),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.d_model, self.model_config.d_model),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.d_model, self.model_config.d_model * self.model_config.num_hidden_layers * 2)
        )
        
        self.attn_layer = nn.MultiheadAttention(self.config.latent_dim, num_heads=4, dropout=0.2)

    def forward(self, source_embedding, embed_query):
        batch_size = source_embedding[0].size(0)
        source_value, source_attention_mask = source_embedding
        # attention layer between AMR [batch_size, max_graph_len, 768] and tokens [batch_size, prefix_length, 768]
        embed_query, source_value = [x.transpose(1,0) for x in [embed_query, source_value]]
        key_padding_mask = (1 - source_attention_mask).bool() # True indicates ignore

        attn_output = self.attn_layer(embed_query, source_value, source_value, key_padding_mask=key_padding_mask)[0].transpose(1,0)
        
        # [batch_size, prefix_length, 768] project to num_layers, key & value
        prefix_weights = (self.projector(attn_output).view(batch_size, self.config.prefix_length, self.model_config.num_hidden_layers, -1)).transpose(1, 2)

        return prefix_weights

class LinearIndepProjector(nn.Module):
    """
    This is projector pass EACH embed_query's through linear layers
    """
    def __init__(self, config, model_config):
        super().__init__()
        self.config = config
        self.model_config = model_config
        logger.info(f'Using projector {self.__class__.__name__}')
        prefix_final_dim = self.model_config.num_hidden_layers * self.model_config.d_model * 2
        self.projector = nn.Sequential(
            nn.Linear(self.config.project_hidden_dim, self.config.project_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(self.config.project_hidden_dim, self.config.project_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(self.config.project_hidden_dim, prefix_final_dim)
        )

    def forward(self, _, embed_query):
        batch_size = embed_query.size(0)
        prefix_length = embed_query.size(1)
        prefix_weights = self.projector(embed_query)
        prefix_weights = (prefix_weights.view(batch_size, prefix_length, self.model_config.num_hidden_layers, -1)).transpose(1, 2)

        return prefix_weights
