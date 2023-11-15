## Query and Extract

We implement the model from [Query and Extract: Refining Event Extraction as Type-oriented Binary Decoding](https://arxiv.org/abs/2110.07476). This model suports event detection and event argument extraction. The code is adapted from the original [codebase](https://github.com/VT-NLP/Event_Query_Extract). 

Since the origin model supports event argument role labeling rather than event argument extraction, we learn an additional NER model during training and use the predicted entities for event argument role labeling during testing.

The original codebase has several bugs and we made some major changes of EARL model:
* Added implementation using RoBERTa as pretrained model.
* Changed data preprocessing and evaluation code to align with our framework.
* Use gold entities to perform EARL task.
* Changed normalization method in EARLmodel. (tag: `[Change]`)
* Added implementation of trigger_aware entity embedding as described in original paper. (tag: `[Change]`)
* Changed the implementation method of creating entity_mapping matrix, to adapt to argument overlapping scenario.

```bib
@inproceedings{Wang22queryextract,
  author       = {Sijia Wang and
                  Mo Yu and
                  Shiyu Chang and
                  Lichao Sun and
                  Lifu Huang},
  title        = {Query and Extract: Refining Event Extraction as Type-oriented Binary
                  Decoding},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2022},
  year         = {2022},
}
```
