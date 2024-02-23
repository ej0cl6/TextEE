local task = "ED";
local dataset = "genia2013";
local split = 1;
local model_type = "CEDAR";
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
    "max_length": 500, 
    
    // TI model config
    "ti_base_model_dropout": 0.2,
    "ti_use_crf": true,
    "ti_type_feature_num": 100, 
    "ti_linear_hidden_num": 150,
    "ti_linear_dropout": 0.2,
    "ti_linear_bias": true, 
    "ti_linear_activation": "relu",
    "ti_multi_piece_strategy": "average", 
    
    // TI train config
    "ti_max_epoch": 30,
    "ti_warmup_epoch": 5,
    "ti_accumulate_step": 1,
    "ti_train_batch_size": 12,
    "ti_eval_batch_size": 12,
    "ti_learning_rate": 0.001,
    "ti_base_model_learning_rate": 1e-05,
    "ti_weight_decay": 0.001,
    "ti_base_model_weight_decay": 1e-05,
    "ti_grad_clipping": 5.0,
    
    // ETR model config
    "etr_linear_hidden_num": 128,
    "etr_margin": 1.0,
    "etr_max_pos": 8, 
    "etr_max_neg": 8, 
    "etr_max_select": 8, 
    
    // ETR train config
    "etr_max_epoch": 30,
    "etr_warmup_epoch": 5,
    "etr_accumulate_step": 1,
    "etr_train_batch_size": 12,
    "etr_eval_batch_size": 12,
    "etr_learning_rate": 0.001,
    "etr_base_model_learning_rate": 1e-05,
    "etr_weight_decay": 0.001,
    "etr_base_model_weight_decay": 1e-05,
    "etr_grad_clipping": 5.0,
    
    // ETC model config
    "etc_n_neg": 2, 
    
    // ETC train config
    "etc_max_epoch": 30,
    "etc_warmup_epoch": 5,
    "etc_accumulate_step": 1,
    "etc_train_batch_size": 12,
    "etc_eval_batch_size": 12,
    "etc_learning_rate": 0.001,
    "etc_base_model_learning_rate": 1e-05,
    "etc_weight_decay": 0.001,
    "etc_base_model_weight_decay": 1e-05,
    "etc_grad_clipping": 5.0,
    

}
