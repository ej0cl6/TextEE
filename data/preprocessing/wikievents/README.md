## WikiEvents

The WikiEvents dataset comes from NAACL 2021 paper [Document-Level Event Argument Extraction by Conditional Generation](https://arxiv.org/abs/2104.05919). WikiEvents is an event argument extraction dataset.

### Preprocessing Steps

- Download dataset from
  - [https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl](https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl)
  - [https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl](https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl)
  - [https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl](https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl)
- Set `WIKIEVENTS_PATH` in `process_wikievents.sh`
- Run `process_wikievents.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  197  |    450     |      50      |       3131       |      57     |    4393    |
| Dev   |   24  |     53     |      39      |       422        |      43     |     592    |
| Test  |   24  |     62     |      38      |       379        |      46     |     516    |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  197  |    439     |      50      |       2990       |      57     |    4234    |
| Dev   |   24  |     57     |      39      |       405        |      42     |     571    |
| Test  |   24  |     69     |      37      |       537        |      38     |     696    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  197  |    435     |      50      |       3014       |      56     |    4228    |
| Dev   |   24  |     78     |      36      |       471        |      43     |     623    |
| Test  |   24  |     52     |      37      |       447        |      47     |     650    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  197  |    454     |      50      |       3143       |      57     |    4391    |
| Dev   |   24  |     46     |      36      |       431        |      43     |     606    |
| Test  |   24  |     65     |      40      |       358        |      47     |     504    |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  197  |    441     |      50      |       3142       |      57     |    4370    |
| Dev   |   24  |     57     |      38      |       394        |      43     |     562    |
| Test  |   24  |     67     |      40      |       396        |      45     |     569    |

### Citation

```bib
@inproceedings{Li21wikievents,
  author       = {Sha Li and
                  Heng Ji and
                  Jiawei Han},
  title        = {Document-Level Event Argument Extraction by Conditional Generation},
  booktitle    = {Proceedings of the 2021 Conference of the North American Chapter of
                  the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year         = {2021},
}
```

