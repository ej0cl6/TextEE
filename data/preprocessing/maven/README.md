## MAVEN

The MAVEN dataset comes from EMNLP 2020 paper [MAVEN: A Massive General Domain Event Detection Dataset](https://arxiv.org/abs/2004.13590). MAVEN targets the event detection task only. Since the testing ground truths are not public, our splits are generated with the training set and the validation set.

### Preprocessing Steps

- Download `train.jsonl` and `valid.jsonl`from [https://github.com/THU-KEG/MAVEN-dataset/](https://github.com/THU-KEG/MAVEN-dataset/) and set `MAVEN_PATH` in `process_maven.sh`
- Run `process_maven.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  2537 |    28734   |      168     |       69069      |
| Dev   |  543  |    5814    |      167     |       13638      |
| Test  |  543  |    5925    |      168     |       14190      |

Split2
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  2537 |    28341   |      168     |       68162      |
| Dev   |  543  |    5982    |      167     |       14233      |
| Test  |  543  |    6150    |      168     |       14502      |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  2537 |    28348   |      168     |       67832      |
| Dev   |  543  |    6049    |      167     |       14185      |
| Test  |  543  |    6076    |      168     |       14880      |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  2537 |    28172   |      168     |       67450      |
| Dev   |  543  |    6190    |      167     |       14637      |
| Test  |  543  |    6111    |      167     |       14810      |

Split5
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  2537 |    28261   |      168     |       67826      |
| Dev   |  543  |    6190    |      167     |       14493      |
| Test  |  543  |    6022    |      168     |       14578      |

### Citation

```bib
@inproceedings{wang2020MAVEN,
    title = {{MAVEN}: A Massive General Domain Event Detection Dataset},
    author = {Wang, Xiaozhi and Wang, Ziqi and Han, Xu and Jiang, Wangyi and Han, Rong and Liu, Zhiyuan and Li, Juanzi and Li, Peng and Lin, Yankai and Zhou, Jie},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2020},
}
```

