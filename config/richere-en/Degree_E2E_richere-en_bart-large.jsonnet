local task = "E2E";
local dataset = "richere-en";
local split = 1;
local model_type = "Degree";
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
    
    // model config
    "pretrained_model_name": pretrained_model_name,
    "input_style": ["event_type_sent", "keywords", "template"], 
    "output_style": ["trigger:sentence", "argument:sentence"], 
    "n_negative": 15, 
    "max_length": 450,
    "max_output_length": 100,
    
    // train config
    "max_epoch": 45,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "train_batch_size": 24,
    "eval_batch_size": 32,
    "learning_rate": 1e-05,
    "weight_decay": 1e-05,
    "grad_clipping": 5.0,
    "beam_size": 4,
}
