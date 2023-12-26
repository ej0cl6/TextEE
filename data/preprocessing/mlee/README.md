## MLEE

The MLEE dataset comes from Bioinformatics 2012 paper [Event extraction across multiple levels of biological organization](https://academic.oup.com/bioinformatics/article/28/18/i575/249872). MLEE is an end-to-end event extraction dataset.

### Preprocessing Steps

- Download dataset from [https://www.nactem.ac.uk/MLEE/MLEE-1.0.2-rev1.tar.gz](https://www.nactem.ac.uk/MLEE/MLEE-1.0.2-rev1.tar.gz) and uncompress it
- Set `MLEE_PATH` in `process_mlee.sh`
- Run `process_mlee.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  184  |     199    |      29      |       4705       |      14     |    4237    |
| Dev   |   39  |      45    |      21      |       1003       |       9     |     895    |
| Test  |   39  |      42    |      21      |        867       |      12     |     826    |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  184  |     202    |      29      |       4733       |      14     |    4258    |
| Dev   |   39  |      42    |      19      |        898       |      10     |     854    |
| Test  |   39  |      42    |      21      |        944       |      11     |     846    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  184  |     200    |      29      |       4627       |      14     |    4165    |
| Dev   |   39  |      42    |      20      |       1029       |      10     |     944    |
| Test  |   39  |      44    |      20      |        919       |      10     |     849    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  184  |     203    |      29      |       4629       |      14     |    4236    |
| Dev   |   39  |      40    |      20      |        980       |      11     |     872    |
| Test  |   39  |      43    |      20      |        966       |      11     |     850    |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  184  |     201    |      29      |       4653       |      14     |    4200    |
| Dev   |   39  |      42    |      21      |        887       |      11     |     843    |
| Test  |   39  |      43    |      20      |       1035       |      11     |     915    |

### Citation

```bib
@article{Pyysalo12mlee,
  author       = {Sampo Pyysalo and
                  Tomoko Ohta and
                  Makoto Miwa and
                  Han{-}Cheol Cho and
                  Junichi Tsujii and
                  Sophia Ananiadou},
  title        = {Event extraction across multiple levels of biological organization},
  journal      = {Bioinformatics},
  volume       = {28},
  number       = {18},
  pages        = {575--581},
  year         = {2012},
}
```

