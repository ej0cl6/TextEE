## Genia2013

The Genia2013 dataset comes from BioNLP Shared Task 2013 Workshop [The Genia Event Extraction Shared Task, 2013 Edition - Overview](https://aclanthology.org/W13-2002/). Genia2013 is an end-to-end event extraction dataset.

### Preprocessing Steps

- Download `BioNLP-ST-2013_GE_train_data_rev3.tar.gz` and `BioNLP-ST-2013_GE_devel_data_rev3.tar.gz` from [https://2013.bionlp-st.org/tasks](https://2013.bionlp-st.org/tasks) and uncompress them
- Set `GENIA_PATH` in `process_genia.sh`
- Run `process_genia.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |   12  |     420    |      13      |       4077       |       7     |    3921    |
| Dev   |   4   |     105    |      10      |        950       |       7     |     858    |
| Test  |   4   |     139    |      11      |        974       |       7     |     881    |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |   12  |     388    |      13      |       3578       |       7     |    3561    |
| Dev   |   4   |     128    |      11      |       1284       |       6     |    1134    |
| Test  |   4   |     148    |      10      |       1149       |       6     |     965    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |   12  |     381    |      13      |       3816       |       7     |    3674    |
| Dev   |   4   |     143    |      10      |       1174       |       7     |    1079    |
| Test  |   4   |     140    |      11      |       1011       |       6     |     907    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |   12  |     441    |      13      |       3971       |       7     |    3993    |
| Dev   |   4   |     111    |       9      |        785       |       7     |     616    |
| Test  |   4   |     112    |      11      |       1245       |       6     |    1051    |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |   12  |     427    |      13      |       4225       |       7     |    4112    |
| Dev   |   4   |     120    |      10      |        809       |       6     |     717    |
| Test  |   4   |     117    |      10      |        967       |       7     |     831    |

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

