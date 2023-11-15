local task = "E2E";
local dataset = "casie";
local split = 1;
local model_type = "AMRIE";
local pretrained_model_name = "roberta-large";
local pretrained_model_alias = {
    "roberta-large": "roberta-large", 
};

{
    // general config
    "task": task, 
    "dataset": dataset,
    "model_type": model_type, 
    "gpu_device": 0, 
    "seed": 0, 
    "cache_dir": "./cache", 
    "output_dir": "./outputs/%s_%s_%s_split%s_%s" % [model_type, task, dataset, split, pretrained_model_alias[pretrained_model_name]], 
    "train_file": "./data/processed_data/%s/split%s/train.json" % [dataset, split],
    "dev_file": "./data/processed_data/%s/split%s/dev.json" % [dataset, split],
    "test_file": "./data/processed_data/%s/split%s/test.json" % [dataset, split],
    
    // resource config
    "valid_pattern_path": "./TextEE/models/AMRIE/valid_patterns/valid_patterns_%s/" % [dataset], 
    "processed_train_amr": "./TextEE/models/AMRIE/processed_amr/%s/split%s/train_graphs.pkl" % [dataset, split], 
    "processed_dev_amr": "./TextEE/models/AMRIE/processed_amr/%s/split%s/dev_graphs.pkl" % [dataset, split], 
    "processed_test_amr": "./TextEE/models/AMRIE/processed_amr/%s/split%s/test_graphs.pkl" % [dataset, split], 
    
    // model config
    "pretrained_model_name": pretrained_model_name,
    "max_length": 510,
    "multi_piece_strategy": "average",
    "bert_dropout": 0.5,
    "use_extra_bert": true,
    "extra_bert": -3,
    "use_global_features": true,
    "global_features": [],
    "global_warmup": 0,
    "linear_dropout": 0.4,
    "linear_bias": true,
    "linear_activation": "relu", 
    "entity_hidden_num": 150,
    "mention_hidden_num": 150,
    "event_hidden_num": 600,
    "relation_hidden_num": 150,
    "role_hidden_num": 600,
    "use_entity_type": true,
    "beam_size": 20,
    "beta_v": 2,
    "beta_e": 2,
    "relation_mask_self": true,
    "relation_directional": false,
    "symmetric_relations": [],
    "edge_type_num": 13,
    "edge_type_dim": 256,
    "gnn_layers": 2,
    "lamda": 0.002,
    "sort_by_amr": true,
    "use_graph_encoder": true,
    "target_task": "role",
    
    // train config
    "max_epoch": 100,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "batch_size": 4,
    "eval_batch_size": 8,
    "learning_rate": 1e-3,
    "bert_learning_rate": 1e-05,
    "weight_decay": 1e-3,
    "bert_weight_decay": 1e-5,
    "grad_clipping": 5.0,
    
}
