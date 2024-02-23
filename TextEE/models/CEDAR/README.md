## CEDAR

We implement the model from [GLEN: General-Purpose Event Detection for Thousands of Types](https://arxiv.org/abs/2303.09093). This model suports event detection. The code is adapted from the original [codebase](https://github.com/ZQS1943/GLEN). 

Notice that the original paper considers self-labeling during training as the dataset they consider is noisy. Our implementation currently ignores the self-labeling part.


```bib
@inproceedings{Li23cedar,
  author       = {Sha Li and
                  Qiusi Zhan and
                  Kathryn Conger and
                  Martha Palmer and
                  Heng Ji and
                  Jiawei Han},
  title        = {{GLEN:} General-Purpose Event Detection for Thousands of Types},
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year         = {2023},
}
```
