## BART-Gen

We implement the model from [Document-Level Event Argument Extraction by Conditional Generation](https://arxiv.org/abs/2104.05919). This model suports event argument extraction. The code is adapted from the original [codebase](https://github.com/raspberryice/gen-arg). Notice that we replace the original pure copy mechanism with copy-generator since we observe this works better.

```bib
@inproceedings{Li21bartgen,
  author       = {Sha Li and
                  Heng Ji and
                  Jiawei Han},
  title        = {Document-Level Event Argument Extraction by Conditional Generation},
  booktitle    = {Proceedings of the 2021 Conference of the North American Chapter of
                  the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year         = {2021},
}
```
