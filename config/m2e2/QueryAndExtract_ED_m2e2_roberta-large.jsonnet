local task = "ED";
local dataset = "m2e2";
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
    "pattern_path": "./TextEE/models/QueryAndExtract/patterns.json",

    // model config
    "trigger_threshold": 0.65,
    "sampling": 0.5,
    "pretrained_model_name": pretrained_model_name,
    "non_weight": 1,
    "trigger_training_weight": 1,
    "last_k_hidden": 3,
    "use_pos_tag": true,
    "max_length": 200, 
    
    // train config
    "max_epoch": 30,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "train_batch_size": 8,
    "learning_rate": 1e-4,
    "base_model_learning_rate": 1e-05,
    "dropout": 0.5,
    "weight_decay": 1e-3,
    "grad_clipping": 5.0,
    "eps": 1e-6,
}
