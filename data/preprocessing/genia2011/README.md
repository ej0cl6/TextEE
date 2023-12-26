## Genia2011

The Genia2011 dataset comes from BioNLP Shared Task 2011 Workshop [Overview of Genia Event Task in BioNLP Shared Task 2011](https://aclanthology.org/W11-1802/). Genia2011 is an end-to-end event extraction dataset.

### Preprocessing Steps

- Download `BioNLP-ST_2011_genia_train_data_rev1.tar.gz` and `BioNLP-ST_2011_genia_devel_data_rev1.tar.gz` from [https://bionlp-st.dbcls.jp/GE/2011/downloads/](https://bionlp-st.dbcls.jp/GE/2011/downloads/) and uncompress them
- Set `GENIA_PATH` in `process_genia.sh`
- Run `process_genia.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  576  |     773    |       9      |       7396       |      10     |    6495    |
| Dev   |  192  |     348    |       9      |       3773       |       9     |    3352    |
| Test  |  192  |     254    |       9      |       2368       |       8     |    2018    |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  576  |     843    |       9      |       8455       |      10     |    7397    |
| Dev   |  192  |     266    |       9      |       2713       |       9     |    2358    |
| Test  |  192  |     266    |       9      |       2369       |       9     |    2110    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  576  |     901    |       9      |       8638       |      10     |    7687    |
| Dev   |  192  |     233    |       9      |       2042       |       8     |    1743    |
| Test  |  192  |     241    |       9      |       2857       |       9     |    2435    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  576  |     808    |       9      |       7836       |      10     |    7037    |
| Dev   |  192  |     277    |       9      |       2842       |       9     |    2319    |
| Test  |  192  |     290    |       9      |       2859       |       9     |    2509    |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  576  |     853    |       9      |       8460       |      10     |    7464    |
| Dev   |  192  |     240    |       9      |       2368       |       9     |    2061    |
| Test  |  192  |     282    |       9      |       2709       |       9     |    2340    |

### Citation

```bib
@inproceedings{Kim11genia2011,
  author       = {Jin{-}Dong Kim and
                  Yue Wang and
                  Toshihisa Takagi and
                  Akinori Yonezawa},
  title        = {Overview of Genia Event Task in BioNLP Shared Task 2011},
  booktitle    = {Proceedings of BioNLP Shared Task 2011 Workshop},
  year         = {2011},
}
```

