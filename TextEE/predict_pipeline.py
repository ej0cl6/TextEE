import os, logging, json, pprint
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, load_text, save_predictions, set_gpu, parse_unknown_args, convert_ED_to_EAE, combine_ED_and_EAE_to_E2E
import ipdb

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--ed_model', required=True)
    parser.add_argument('--eae_model', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown_args(unknown_args)
    
    set_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
    
    # load trainer
    assert args.ed_model
    ed_config = load_config(os.path.join(args.ed_model, "config.json"))
    ed_config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(ed_config), indent=4)}")
    ed_trainer_class = TRAINER_MAP[(ed_config.model_type, ed_config.task)]

    # load data
    ed_eval_data, eval_offset_map = load_text("ED", args.input_file, ed_trainer_class.add_extra_info_fn, ed_config)

    # load model
    ed_trainer = ed_trainer_class(ed_config)
    ed_trainer.load_model(checkpoint=args.ed_model)

    # ED predict 
    ed_predictions = ed_trainer.predict(ed_eval_data, **unknown_args)
    eae_eval_data = convert_ED_to_EAE(ed_predictions, ed_eval_data)
    
    # load EAE trainer and model
    eae_config = load_config(os.path.join(args.eae_model, "config.json"))
    eae_config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(eae_config), indent=4)}")
    eae_trainer_class = TRAINER_MAP[(eae_config.model_type, eae_config.task)]
    eae_trainer = eae_trainer_class(eae_config)
    eae_trainer.load_model(checkpoint=args.eae_model)
    
    # EAE predict 
    eae_predictions = eae_trainer.predict(eae_eval_data, **unknown_args)
    e2e_predictions = combine_ED_and_EAE_to_E2E(ed_predictions, eae_predictions)
    
    # save predictions
    save_predictions(args.output_file, e2e_predictions, ed_eval_data, eval_offset_map)
    
if __name__ == "__main__":
    main()
    
    
