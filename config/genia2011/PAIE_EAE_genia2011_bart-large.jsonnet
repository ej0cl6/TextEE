local task = "EAE";
local dataset = "genia2011";
local split = 1;
local model_type = "PAIE";
local pretrained_model_name = "facebook/bart-large";
local pretrained_model_alias = {
    "facebook/bart-large": "bart-large", 
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
    "prompt_path": "./TextEE/models/PAIE/patterns/prompts_%s_full.csv" % [dataset],
    "role_path": "./TextEE/models/PAIE/patterns/description_%s.csv" % [dataset],
    
    // model config
    "pretrained_model_name": pretrained_model_name,
    "context_representation": "decoder",
    "matching_method_train": "max",
    "max_span_length": 10,
    "bipartite": false,
    "pad_mask_token": 0,
    "max_enc_seq_length": 510,
    "max_prompt_seq_length": 120,
    "max_length": 500,
    
    
    // train config
    "max_epoch": 90,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "train_batch_size": 3,
    "eval_batch_size": 3,
    "learning_rate": 2e-05,
    "weight_decay": 0.01,
    "grad_clipping": 5,
    "adam_epsilon": 1e-8,
}
