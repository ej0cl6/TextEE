import os, logging, json, pprint
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, load_text, save_predictions, set_gpu, parse_unknown_args
import ipdb

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown_args(unknown_args)
    
    set_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
    
    config = load_config(os.path.join(args.model, "config.json"))
    config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
    
    # load trainer
    trainer_class = TRAINER_MAP[(config.model_type, config.task)]
    
    # load data
    eval_data, eval_offset_map = load_text("E2E", args.input_file, trainer_class.add_extra_info_fn, config)

    # load model
    assert args.model
    trainer = trainer_class(config)
    trainer.load_model(checkpoint=args.model)
    
    # predict and save predictions
    predictions = trainer.predict(eval_data, **unknown_args)
    save_predictions(args.output_file, predictions, eval_data, eval_offset_map)

        
if __name__ == "__main__":
    main()
    
    
