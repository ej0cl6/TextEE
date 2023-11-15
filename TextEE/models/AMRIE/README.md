## AMR-IE

We implement the model from [Abstract Meaning Representation Guided Graph Encoding and Decoding for Joint Information Extraction](https://aclanthology.org/2021.naacl-main.4/). This model suports end-to-end event extraction. The code is adapted from the original [codebase](https://github.com/zhangzx-uiuc/AMR-IE).

You may need to use the `./process_amr.py` script in the original [codebase](https://github.com/zhangzx-uiuc/AMR-IE) to generate AMR graphs before runing the training sciprt. You can refer [here](https://github.com/zhangzx-uiuc/AMR-IE#datasets) for more information.

Please Remember to modify the corresponding config accordingly. One example is [here](https://github.com/ej0cl6/TextEE/blob/master/config/ace05-en/AMRIE_E2E_ace05-en_roberta-large.jsonnet#L23). 

```bib
@inproceedings{Zhang21amrie,
  author    = {Zixuan Zhang and Heng Ji},
  title     = {Abstract Meaning Representation Guided Graph Encoding and Decoding for Joint Information Extraction},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               (NAACL-HLT)},
  year      = {2021},
}
```
