## CASIE

The CASIE dataset comes from AAAI 2020 paper [CASIE: Extracting Cybersecurity Event Information from Text](https://ojs.aaai.org/index.php/AAAI/article/view/6401). CASIE is an end-to-end event extraction dataset.

### Preprocessing Steps

- Clone the github repo [https://github.com/Ebiquity/CASIE](https://github.com/Ebiquity/CASIE)
- Set `CASIE_PATH` in `process_casie.sh`
- Run `process_casie.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  701  |    1047    |      5       |       5980       |      26     |    15869   |
| Dev   |  149  |    218     |      5       |       1221       |      26     |    3175    |
| Test  |  149  |    218     |      5       |       1268       |      26     |    3531    |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  701  |    1046    |      5       |       6010       |      26     |    15986   |
| Dev   |  149  |    223     |      5       |       1294       |      26     |    3492    |
| Test  |  149  |    214     |      5       |       1165       |      26     |    3097    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  701  |    1044    |      5       |       6009       |      26     |    16090   |
| Dev   |  149  |    210     |      5       |       1286       |      26     |    3344    |
| Test  |  149  |    229     |      5       |       1174       |      26     |    3141    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  701  |    1040    |      5       |       6034       |      26     |    15962   |
| Dev   |  149  |    229     |      5       |       1172       |      26     |    3211    |
| Test  |  149  |    214     |      5       |       1263       |      26     |    3402    |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  701  |    1043    |      5       |       5831       |      26     |    15544   |
| Dev   |  149  |    218     |      5       |       1288       |      26     |    3369    |
| Test  |  149  |    222     |      5       |       1350       |      26     |    3662    |

### Citation

```bib
@inproceedings{Satyapanich20casie,
  author       = {Taneeya Satyapanich and
                  Francis Ferraro and
                  Tim Finin},
  title        = {{CASIE:} Extracting Cybersecurity Event Information from Text},
  booktitle    = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence (AAAI)},
  year         = {2020},
}
```

