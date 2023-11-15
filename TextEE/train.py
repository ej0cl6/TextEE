import os, logging, json
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, load_all_data, set_seed, set_gpu, set_logger
from scorer import compute_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('--seed', type=int, required=False)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)
    set_gpu(config.gpu_device)
    config = set_logger(config)
    
    # load trainer
    trainer_class = TRAINER_MAP[(config.model_type, config.task)]
    
    # load data
    train_data, dev_data, test_data, type_set = load_all_data(config, trainer_class.add_extra_info_fn)
    
    # train
    trainer = trainer_class(config, type_set)
    trainer.train(train_data, dev_data, **vars(config))
    logger.info("Training was done!")
    
    # test
    logger.info("Loading best model for evaluation.")
    trainer.load_model(checkpoint=config.output_dir)
    predictions = trainer.predict(test_data, **vars(config))
    scores = compute_scores(predictions, test_data, config.task)
    print("Test")
    print_scores(scores)
        
if __name__ == "__main__":
    main()
    
    
