## RAMS

The RAMS dataset comes from ACL 2020 paper [Multi-Sentence Argument Linking](https://arxiv.org/abs/1911.03766). RAMS is an event argument extraction dataset.

### Preprocessing Steps

- Download dataset from [https://nlp.jhu.edu/rams/RAMS_1.0c.tar.gz](https://nlp.jhu.edu/rams/RAMS_1.0c.tar.gz), uncompress the file, and set `RAMS_PATH` in `process_rams.sh`
- Run `process_rams.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  7827 |    7827    |     139      |       7287       |      65     |    16951   |
| Dev   |   910 |    910     |     136      |       910        |      64     |    2132    |
| Test  |   910 |    910     |     135      |       910        |      63     |    2123    |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  7827 |    7827    |     139      |       7287       |      65     |    16946   |
| Dev   |   910 |    910     |     135      |       910        |      65     |    2113    |
| Test  |   910 |    910     |     137      |       910        |      65     |    2147    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  7827 |    7827    |     139      |       7287       |      65     |    16937   |
| Dev   |   910 |    910     |     135      |       910        |      64     |    2168    |
| Test  |   910 |    910     |     135      |       910        |      64     |    2101    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  7827 |    7827    |     139      |       7287       |      65     |    17014   |
| Dev   |   910 |    910     |     136      |       910        |      62     |    2093    |
| Test  |   910 |    910     |     137      |       910        |      63     |    2099    |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  7827 |    7827    |     139      |       7287       |      65     |    17003   |
| Dev   |   910 |    910     |     135      |       910        |      63     |    2130    |
| Test  |   910 |    910     |     137      |       910        |      65     |    2073    |

### Citation

```bib
@inproceedings{Ebner20rams,
  author       = {Seth Ebner and
                  Patrick Xia and
                  Ryan Culkin and
                  Kyle Rawlins and
                  Benjamin Van Durme},
  title        = {Multi-Sentence Argument Linking},
  booktitle    = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year         = {2020},
}
```

