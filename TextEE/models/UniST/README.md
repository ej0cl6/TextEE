## UniST

We implement the model from [Unified Semantic Typing with Meaningful Label Inference](https://arxiv.org/abs/2205.01826). This model suports event detection. The code is adapted from the original [codebase](https://github.com/luka-group/unist). 

Since the origin model supports semantic typing only, we learn an additional span recognition model during training and use the predicted trigger spans for trigger span typing during testing.


```bib
@inproceedings{Huang22unist,
  author       = {James Y. Huang and
                  Bangzheng Li and
                  Jiashu Xu and
                  Muhao Chen},
  title        = {Unified Semantic Typing with Meaningful Label Inference},
  booktitle    = {Proceedings of the 2022 Conference of the North American Chapter of
                  the Association for Computational Linguistics: Human Language Technologies (NAACL)},
  year         = {2022},
}
```
