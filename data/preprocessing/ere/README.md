## ERE

The ERE dataset comes from the paper [From Light to Rich ERE: Annotation of Entities, Relations, and Events](https://aclanthology.org/W15-0812/). ERE is an end-to-end event extraction dataset. There are two versions of ERE data: light and rich. Currently, we consider the rich ERE.

### Preprocessing Steps

- Download dataset from [https://catalog.ldc.upenn.edu/LDC2023T04](https://catalog.ldc.upenn.edu/LDC2023T04), uncompress the file, and set `ERE_EN_PATH` in `process_ERE_en.sh`
- Run `process_ERE_en.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  232  |    9198    |     38       |      4549        |      21     |    6581    |
| Dev   |   28  |     876    |     35       |       488        |      21     |     737    |
| Test  |   28  |    1167    |     34       |       672        |      21     |     936    |

Split2
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  232  |    8886    |     38       |      4444        |      21     |    6520    |
| Dev   |   28  |    1299    |     36       |       688        |      21     |     978    |
| Test  |   28  |    1056    |     37       |       577        |      21     |     756    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  232  |    9094    |     38       |      4490        |      21     |    6517    |
| Dev   |   28  |    1081    |     36       |       678        |      21     |     942    |
| Test  |   28  |    1066    |     35       |       541        |      21     |     795    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  232  |    9105    |     38       |      4541        |      21     |    6647    |
| Dev   |   28  |     973    |     34       |       571        |      21     |     804    |
| Test  |   28  |    1163    |     37       |       597        |      21     |     803    |

Split5
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  232  |    9169    |     38       |      4682        |      21     |    6756    |
| Dev   |   28  |    1135    |     34       |       487        |      21     |     692    |
| Test  |   28  |     937    |     35       |       540        |      21     |     806    |

### Citation

```bib
@inproceedings{Song15ere,
  author       = {Zhiyi Song and
                  Ann Bies and
                  Stephanie M. Strassel and
                  Tom Riese and
                  Justin Mott and
                  Joe Ellis and
                  Jonathan Wright and
                  Seth Kulick and
                  Neville Ryant and
                  Xiaoyi Ma},
  title        = {From Light to Rich {ERE:} Annotation of Entities, Relations, and Events},
  booktitle    = {Proceedings of the The 3rd Workshop on {EVENTS:} Definition, Detection, Coreference, and Representation, EVENTS@HLP-NAACL},
  year         = {2015},
}
```
