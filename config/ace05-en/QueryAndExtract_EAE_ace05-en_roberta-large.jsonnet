local task = "EAE";
local dataset = "ace05-en";
local split = 1;
local model_type = "QueryAndExtract";
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
    "metadata_path": "./TextEE/models/QueryAndExtract/metadata.json",

    // model config
    "pretrained_model_name": pretrained_model_name,
    "extra_bert": -3,
    "use_extra_bert": false,
    "n_hid": 768,
    "dropout": 0.5,
    "non_weight": 1,
    "max_length": 175, 
    "max_entity_count": 40,
    
    // train config
    "max_epoch": 90,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "train_batch_size": 4,
    "eval_batch_size": 4,
    "learning_rate": 2e-4,
    "base_model_learning_rate": 1e-05,
    "weight_decay": 0.01,
    "base_model_weight_decay": 1e-03,
    "grad_clipping": 5.0,
    "eps": 1e-6,
    
    // NER model config
    "ner_base_model_dropout": 0.2,
    "ner_linear_hidden_num": 150,
    "ner_linear_dropout": 0.2,
    "ner_linear_bias": true, 
    "ner_linear_activation": "relu",
    "ner_use_crf": true,
    "ner_multi_piece_strategy": "average", 
    
    // NER train config
    "ner_learning_rate": 0.001,
    "ner_base_model_learning_rate": 1e-05,
    "ner_weight_decay": 0.001,
    "ner_base_model_weight_decay": 1e-05,
    "ner_grad_clipping": 5.0,
}
