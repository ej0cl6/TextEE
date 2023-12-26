## ACE05

The ACE05 dataset comes from LREC 2004 paper [The Automatic Content Extraction (ACE) Program - Tasks, Data, and Evaluation](https://aclanthology.org/L04-1011/). ACE05 is an end-to-end event extraction dataset.

### Preprocessing Steps

- Download dataset from [https://catalog.ldc.upenn.edu/LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06), uncompress the file, and set `ACE05_PATH` in `process_ace05_en.sh`
- Run `process_ace05_en.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  481  |   16531    |     33       |      4309        |      22     |    6503    |
| Dev   |   59  |    1870    |     30       |       476        |      22     |     766    |
| Test  |   59  |    2519    |     30       |       563        |      22     |     828    |

Split2
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  481  |   17423    |     33       |      4348        |      22     |    6544    |
| Dev   |   59  |    1880    |     29       |       555        |      22     |     894    |
| Test  |   59  |    1617    |     30       |       445        |      22     |     659    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  481  |   17285    |     33       |      4331        |      22     |    6484    |
| Dev   |   59  |    2123    |     30       |       515        |      22     |     799    |
| Test  |   59  |    1512    |     30       |       502        |      22     |     814    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  481  |   16842    |     33       |      4437        |      22     |    6711    |
| Dev   |   59  |    1979    |     30       |       460        |      22     |     728    |
| Test  |   59  |    2099    |     29       |       451        |      22     |     658    |

Split5
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  481  |   16355    |     33       |      4198        |      22     |    6392    |
| Dev   |   59  |    1933    |     30       |       509        |      22     |     772    |
| Test  |   59  |    2632    |     31       |       641        |      22     |     933    |

### Citation

```bib
@inproceedings{Doddington04ace,
  author       = {George R. Doddington and
                  Alexis Mitchell and
                  Mark A. Przybocki and
                  Lance A. Ramshaw and
                  Stephanie M. Strassel and
                  Ralph M. Weischedel},
  title        = {The Automatic Content Extraction {(ACE)} Program - Tasks, Data, and Evaluation},
  booktitle    = {Proceedings of the Fourth International Conference on Language Resources and Evaluation (LREC)},
  year         = {2004},
}
```
