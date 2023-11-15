import os, logging, json, pprint
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, load_data, set_seed, set_gpu, parse_unknown_args
from scorer import compute_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('--task', required=True, choices=["E2E", "EAE", "EARL"])
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown_args(unknown_args)
    
    set_seed(args.seed)
    set_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
    
    config = load_config(os.path.join(args.model, "config.json"))
    config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
    
    # load trainer
    trainer_class = TRAINER_MAP[(config.model_type, config.task)]
    
    # load data
    eval_data, _ = load_data("E2E", args.data, trainer_class.add_extra_info_fn, config)

    # load model
    assert args.model
    trainer = trainer_class(config)
    trainer.load_model(checkpoint=args.model)
    if args.task == "E2E":
        predictions = trainer.predict(eval_data, **unknown_args)
    elif args.task == "EAE":
        predictions = trainer.predictEAE(eval_data, **unknown_args)
    elif args.task == "EARL":
        predictions = trainer.predictEARL(eval_data, **unknown_args)
    scores = compute_scores(predictions, eval_data, "E2E")
    print("Evaluate")
    print_scores(scores)

        
if __name__ == "__main__":
    main()
    
    
