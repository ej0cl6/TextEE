## PHEE

The PHEE dataset comes from EMNLP 2022 paper [DPHEE: A Dataset for Pharmacovigilance Event Extraction from Text](https://arxiv.org/abs/2210.12560). PHEE is an end-to-end event extraction dataset.

### Preprocessing Steps

- Download dataset from
  - [https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/eeqa/train.json](https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/eeqa/train.json)
  - [https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/eeqa/dev.json](https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/eeqa/dev.json)
  - [https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/eeqa/test.json](https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/eeqa/test.json)
- Set `PHEE_PATH` in `process_phee.sh`
- Run `process_phee.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 2897  |    2897    |       2      |       3003       |      16     |    15482   |
| Dev   |  965  |     965    |       2      |       1011       |      16     |     5123   |
| Test  |  965  |     965    |       2      |       1005       |      16     |     5155   |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 2897  |    2897    |       2      |       3014       |      16     |    15576   |
| Dev   |  965  |     965    |       2      |       1002       |      16     |     5090   |
| Test  |  965  |     965    |       2      |       1003       |      16     |     5094   |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 2897  |    2897    |       2      |       3009       |      16     |    15230   |
| Dev   |  965  |     965    |       2      |       1001       |      16     |     5200   |
| Test  |  965  |     965    |       2      |       1009       |      16     |     5330   |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 2897  |    2897    |       2      |       3020       |      16     |    15496   |
| Dev   |  965  |     965    |       2      |        996       |      16     |     5124   |
| Test  |  965  |     965    |       2      |       1003       |      16     |     5140   |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 2897  |    2897    |       2      |       3011       |      16     |    15498   |
| Dev   |  965  |     965    |       2      |       1000       |      16     |     5049   |
| Test  |  965  |     965    |       2      |       1008       |      16     |     5213   |

### Citation

```bib
@inproceedings{Sun22phee,
  author       = {Zhaoyue Sun and
                  Jiazheng Li and
                  Gabriele Pergola and
                  Byron C. Wallace and
                  Bino John and
                  Nigel Greene and
                  Joseph Kim and
                  Yulan He},
  title        = {{PHEE:} {A} Dataset for Pharmacovigilance Event Extraction from Text},
  booktitle    = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year         = {2022},
}
```

