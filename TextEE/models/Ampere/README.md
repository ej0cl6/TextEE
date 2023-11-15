## AMPERE

We implement the model from [AMPERE: AMR-Aware Prefix for Generation-Based Event Argument Extraction Model](https://arxiv.org/abs/2305.16734). This model suports event argument extraction. The code is adapted from the original [codebase](https://github.com/PlusLabNLP/AMPERE).

You may need to use the following command to generate AMR graphs before runing the training sciprt. You can refer [here](https://github.com/ej0cl6/TextEE/blob/master/TextEE/models/Ampere/generate_amr.sh) for more information.

```
python generate_amr.py -i [input_file] -o [output_amr]
```

Please Remember to modify the corresponding config accordingly. One example is [here](https://github.com/ej0cl6/TextEE/blob/master/config/ace05-en/Ampere_EAE_ace05-en_bart-large.jsonnet#L23). 

```bib
@inproceedings{Hsu23ampere,
  author       = {I{-}Hung Hsu and
                  Zhiyu Xie and
                  Kuan{-}Hao Huang and
                  Prem Natarajan and
                  Nanyun Peng},
  title        = {{AMPERE:} AMR-Aware Prefix for Generation-Based Event Argument Extraction
                  Model},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year         = {2023},
}
```
