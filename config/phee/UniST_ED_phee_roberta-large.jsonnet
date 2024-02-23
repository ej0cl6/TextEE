local task = "ED";
local dataset = "phee";
local split = 1;
local model_type = "UniST";
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
    
    // model config
    "pretrained_model_name": pretrained_model_name,
    "max_length": 200, 
    "margin": 0.1,
    "max_sample_trigger": 20, 
    
    // train config
    "max_epoch": 30,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "train_batch_size": 6,
    "eval_batch_size": 12,
    "learning_rate": 0.001,
    "base_model_learning_rate": 1e-05,
    "weight_decay": 0.001,
    "base_model_weight_decay": 1e-05,
    "grad_clipping": 5.0,
    
    // span model config
    "span_base_model_dropout": 0.2,
    "span_use_crf": true,
    "span_type_feature_num": 100, 
    "span_linear_hidden_num": 150,
    "span_linear_dropout": 0.2,
    "span_linear_bias": true, 
    "span_linear_activation": "relu",
    "span_multi_piece_strategy": "average", 
    
    // span train config
    "span_max_epoch": 30,
    "span_warmup_epoch": 5,
    "span_accumulate_step": 1,
    "span_train_batch_size": 12,
    "span_eval_batch_size": 12,
    "span_learning_rate": 0.001,
    "span_base_model_learning_rate": 1e-05,
    "span_weight_decay": 0.001,
    "span_base_model_weight_decay": 1e-05,
    "span_grad_clipping": 5.0,
}
