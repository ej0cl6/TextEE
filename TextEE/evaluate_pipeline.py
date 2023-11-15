import os, logging, json, pprint
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, load_data, set_seed, set_gpu, parse_unknown_args, convert_ED_to_EAE, combine_ED_and_EAE_to_E2E
from scorer import compute_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('--task', required=True, choices=["ED", "EAE", "E2E", "EARL"])
    parser.add_argument('--data', required=True)
    parser.add_argument('--ed_model', required=False)
    parser.add_argument('--eae_model', required=False)
    parser.add_argument('--earl_model', required=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown_args(unknown_args)
    
    set_seed(args.seed)
    set_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
    
    if args.task == "ED":
        # load trainer
        assert args.ed_model
        ed_config = load_config(os.path.join(args.ed_model, "config.json"))
        ed_config.gpu_device = args.gpu
        logger.info(f"\n{pprint.pformat(vars(ed_config), indent=4)}")
        ed_trainer_class = TRAINER_MAP[(ed_config.model_type, ed_config.task)]
        
        # load data
        eval_data, _ = load_data(args.task, args.data, ed_trainer_class.add_extra_info_fn, ed_config)
        
        # load model
        ed_trainer = ed_trainer_class(ed_config)
        ed_trainer.load_model(checkpoint=args.ed_model)
        
        # predict
        predictions = ed_trainer.predict(eval_data, **unknown_args)
        scores = compute_scores(predictions, eval_data, args.task)
        print("Evaluate")
        print_scores(scores)
        
    elif args.task == "EAE":
        # load trainer
        assert args.eae_model
        eae_config = load_config(os.path.join(args.eae_model, "config.json"))
        eae_config.gpu_device = args.gpu
        logger.info(f"\n{pprint.pformat(vars(eae_config), indent=4)}")
        eae_trainer_class = TRAINER_MAP[(eae_config.model_type, eae_config.task)]
        
        # load data
        eval_data, _ = load_data(args.task, args.data, eae_trainer_class.add_extra_info_fn, eae_config)
        
        # load model
        eae_trainer = eae_trainer_class(eae_config)
        eae_trainer.load_model(checkpoint=args.eae_model)
        
        # predict
        predictions = eae_trainer.predict(eval_data, **unknown_args)
        scores = compute_scores(predictions, eval_data, args.task)
        print("Evaluate")
        print_scores(scores)
        
    elif args.task == "E2E":
        # load ED trainer
        assert args.ed_model and args.eae_model
        ed_config = load_config(os.path.join(args.ed_model, "config.json"))
        ed_config.gpu_device = args.gpu
        logger.info(f"\n{pprint.pformat(vars(ed_config), indent=4)}")
        ed_trainer_class = TRAINER_MAP[(ed_config.model_type, ed_config.task)]
        
        # load data
        ed_eval_data, _ = load_data("ED", args.data, ed_trainer_class.add_extra_info_fn, ed_config)
        gold_data, _ = load_data("E2E", args.data, ed_trainer_class.add_extra_info_fn, ed_config)
        
        # load ED model
        ed_trainer = ed_trainer_class(ed_config)
        ed_trainer.load_model(checkpoint=args.ed_model)
        ed_predictions = ed_trainer.predict(ed_eval_data, **unknown_args)
        eae_eval_data = convert_ED_to_EAE(ed_predictions, ed_eval_data)
        
        # load EAE trainer and model
        eae_config = load_config(os.path.join(args.eae_model, "config.json"))
        eae_config.gpu_device = args.gpu
        logger.info(f"\n{pprint.pformat(vars(eae_config), indent=4)}")
        eae_trainer_class = TRAINER_MAP[(eae_config.model_type, eae_config.task)]
        eae_trainer = eae_trainer_class(eae_config)
        eae_trainer.load_model(checkpoint=args.eae_model)
        eae_predictions = eae_trainer.predict(eae_eval_data, **unknown_args)
        e2e_predictions = combine_ED_and_EAE_to_E2E(ed_predictions, eae_predictions)
        
        scores = compute_scores(e2e_predictions, gold_data, "E2E")
        print("Evaluate")
        print_scores(scores)
        
    elif args.task == "EARL":
        # load trainer
        assert args.earl_model
        earl_config = load_config(os.path.join(args.earl_model, "config.json"))
        earl_config.gpu_device = args.gpu
        logger.info(f"\n{pprint.pformat(vars(earl_config), indent=4)}")
        earl_trainer_class = TRAINER_MAP[(earl_config.model_type, earl_config.task)]
        
        # load data
        eval_data, _ = load_data(args.task, args.data, earl_trainer_class.add_extra_info_fn, earl_config)
        
        # load model
        earl_trainer = earl_trainer_class(earl_config)
        earl_trainer.load_model(checkpoint=args.earl_model)
        
        # predict
        predictions = earl_trainer.predict(eval_data, **unknown_args)
        scores = compute_scores(predictions, eval_data, args.task)
        print("Evaluate")
        print_scores(scores)

        
if __name__ == "__main__":
    main()
    
    
