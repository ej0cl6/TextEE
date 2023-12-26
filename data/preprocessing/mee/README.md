## MEE

The MEE dataset comes from EMNLP 2022 paper [MEE: A Novel Multilingual Event Extraction Dataset](https://arxiv.org/abs/2211.05955). MEE is an end-to-end event extraction dataset.

### Preprocessing Steps

- Download dataset from [http://nlp.uoregon.edu/download/MEE/MEE.zip](http://nlp.uoregon.edu/download/MEE/MEE.zip), uncompress the file, and set `MEE_PATH` in `process_mee.sh`
- Run `process_mee.sh`

### Statistics

#### English

Split1 (English)
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 10400 |    10400   |      16      |       13748      |      23     |    10363   |
| Dev   |  1300 |    1300    |      16      |       1764       |      23     |    1414    |
| Test  |  1300 |    1300    |      16      |       1745       |      23     |    1392    |

Split2 (English)
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 10400 |    10400   |      16      |       13801      |      23     |    10369   |
| Dev   |  1300 |    1300    |      16      |       1731       |      23     |    1614    |
| Test  |  1300 |    1300    |      16      |       1725       |      23     |    1186    |

Split3 (English)
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 10400 |    10400   |      16      |       13847      |      23     |    10803   |
| Dev   |  1300 |    1300    |      16      |       1722       |      23     |    1232    |
| Test  |  1300 |    1300    |      16      |       1688       |      23     |    1134    |

Split4 (English)
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 10400 |    10400   |      16      |       13855      |      23     |    10731   |
| Dev   |  1300 |    1300    |      16      |       1701       |      23     |    1137    |
| Test  |  1300 |    1300    |      16      |       1701       |      23     |    1301    |

Split5 (English)
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train | 10400 |    10400   |      16      |       13802      |      23     |    10595   |
| Dev   |  1300 |    1300    |      16      |       1734       |      23     |    1188    |
| Test  |  1300 |    1300    |      16      |       1721       |      23     |    1386    |

### Citation

```bib
@inproceedings{pouran2022MEE,
    title = {{MEE}: A Novel Multilingual Event Extraction Dataset},
    author = {Pouran Ben Veyseh, Amir and Ebrahimi, Javid and Dernoncourt, Franck and Nguyen, Thien},
    booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2022},
}
```

