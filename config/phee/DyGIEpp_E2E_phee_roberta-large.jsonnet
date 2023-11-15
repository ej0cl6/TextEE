local task = "E2E";
local dataset = "phee";
local split = 1;
local model_type = "DyGIEpp";
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
    "bert_dropout": 0.4,
    "linear_dropout": 0.4,
    "linear_bias": true,
    "linear_activation": "relu",
    "entity_hidden_num": 150,
    "trigger_hidden_num": 150,
    "relation_hidden_num": 150,
    "role_hidden_num": 600,
    "feature_size": 20,
    "max_entity_span": 6,
    "min_entity_span": 1,
    "max_trigger_span": 1,
    "min_trigger_span": 1,
    "ner_loss_weight": 0.5,
    "trigger_loss_weight": 1.0,
    "role_loss_weight": 1.0,
    "relation_loss_weight": 0.0,
    "relation_spans_per_word": 0.0,
    "trigger_spans_per_word": 0.4,
    "argument_spans_per_word": 0.8,
    "target_task": "role",
    
    // train config
    "max_epoch": 60,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "batch_size": 6,
    "eval_batch_size": 12,
    "learning_rate": 0.001,
    "bert_learning_rate": 1e-05,
    "weight_decay": 1e-3,
    "bert_weight_decay": 1e-5,
    "grad_clipping": 5.0,
    
}
