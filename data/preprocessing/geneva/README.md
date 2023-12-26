## GENEVA

The GENEVA dataset comes from ACL 2023 paper [GENEVA: Benchmarking Generalizability for Event Argument Extraction with Hundreds of Event Types and Argument Roles](https://aclanthology.org/2023.acl-long.203.pdf). GENEVA extends MAVEN into the event argument extraction task. Our splits are generated with the complete data released as part of their work.

### Preprocessing Steps

- Download `all_data.json` from https://github.com/PlusLabNLP/GENEVA/blob/main/data and set `GENEVA_PATH` in `process_geneva.sh`
- Run `process_geneva.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  96   |    2582    |      115     |       5290       |     220     |    8618    |
| Dev   |  82   |    509     |      115     |       1016       |     159     |    1683    |
| Test  |  84   |    593     |      115     |       1199       |     171     |    2013    |

Split2
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  97   |    2583    |      115     |       5268       |     220     |    8660    |
| Dev   |  85   |    509     |      114     |       1014       |     158     |    1615    |
| Test  |  85   |    592     |      115     |       1223       |     164     |    1994    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  97   |    2582    |      115     |       5294       |     220     |    8638    |
| Dev   |  85   |    509     |      115     |       1010       |     156     |    1642    |
| Test  |  81   |    593     |      115     |       1201       |     170     |    1989    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  96   |    2582    |      115     |       5293       |     220     |    8705    |
| Dev   |  79   |    509     |      115     |       1003       |     164     |    1636    |
| Test  |  88   |    593     |      115     |       1209       |     166     |    1928    |

Split5
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  97   |    2582    |      115     |       5337       |     220     |    8673    |
| Dev   |  88   |    509     |      115     |       1004       |     161     |    1680    |
| Test  |  86   |    593     |      115     |       1164       |     161     |    1916    |

### Citation

```bib
@inproceedings{parekh-etal-2023-geneva,
    title = "{GENEVA}: Benchmarking Generalizability for Event Argument Extraction with Hundreds of Event Types and Argument Roles",
    author = "Parekh, Tanmay  and
      Hsu, I-Hung  and
      Huang, Kuan-Hao  and
      Chang, Kai-Wei  and
      Peng, Nanyun",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.203",
}
```

