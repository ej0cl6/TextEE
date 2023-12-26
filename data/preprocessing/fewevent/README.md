## FewEvent

The FewEvent dataset comes from WSDM 2020 paper [Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection](https://arxiv.org/abs/1910.11621). FewEvent supports the event detection task only.

### Preprocessing Steps

- Download `Few-Shot_ED.json.zip` from [https://github.com/231sm/Low_Resource_KBP](https://github.com/231sm/Low_Resource_KBP), uncompress it, and set `FEWEVENT_PATH` in `process_fewevent.sh`
- Run `process_fewevent.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  7579 |    7579    |      100     |       7579       |
| Dev   |  2513 |    2513    |      98      |       2513       |
| Test  |  2541 |    2541    |      99      |       2541       |

Split2
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  7579 |    7579    |      100     |       7579       |
| Dev   |  2513 |    2513    |      98      |       2513       |
| Test  |  2541 |    2541    |      99      |       2541       |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  7579 |    7579    |      100     |       7579       |
| Dev   |  2513 |    2513    |      98      |       2513       |
| Test  |  2541 |    2541    |      99      |       2541       |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  7579 |    7579    |      100     |       7579       |
| Dev   |  2513 |    2513    |      98      |       2513       |
| Test  |  2541 |    2541    |      99      |       2541       |

Split5
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  7579 |    7579    |      100     |       7579       |
| Dev   |  2513 |    2513    |      98      |       2513       |
| Test  |  2541 |    2541    |      99      |       2541       |

### Citation

```bib
@inproceedings{Deng20fewevent,
  author       = {Shumin Deng and
                  Ningyu Zhang and
                  Jiaojian Kang and
                  Yichi Zhang and
                  Wei Zhang and
                  Huajun Chen},
  title        = {Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot
                  Event Detection},
  booktitle    = {The Thirteenth {ACM} International Conference on Web Search
                  and Data Mining (WSDM)},
  year         = {2020},
}
```

