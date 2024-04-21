## SPEED

The SPEED dataset comes from NAACL 2024 paper [Event Detection from Social Media for Epidemic Prediction](https://arxiv.org/abs/2404.01679). SPEED is a event detection dataset focusing on edpidemic domain. We use the COVID-only part of SPEED. 

### Preprocessing Steps

- Download `covid-only_speed.json` from https://github.com/PlusLabNLP/SPEED and set `SPEED_PATH` in `process_speed.sh`
- Run `process_speed.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  1185 |    1185    |      7       |       1334       |
| Dev   |  395  |    395     |      7       |       415        |
| Test  |  395  |    395     |      7       |       458        |

Split2
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  1185 |    1185    |      7       |       1361       |
| Dev   |  395  |    395     |      7       |       432        |
| Test  |  395  |    395     |      7       |       424        |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  1185 |    1185    |      7       |       1336       |
| Dev   |  395  |    395     |      7       |       449        |
| Test  |  395  |    395     |      7       |       432        |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  1185 |    1185    |      7       |       1328       |
| Dev   |  395  |    395     |      7       |       460        |
| Test  |  395  |    395     |      7       |       429        |

Split5
|       | #Docs | #Instances | #Event Types | #Events Mentions |
|-------|:-----:|:----------:|:------------:|:----------------:|
| Train |  1185 |    1185    |      7       |       1340       |
| Dev   |  395  |    395     |      7       |       446        |
| Test  |  395  |    395     |      7       |       431        |

### Citation

```bib
@inproceedings{Parekh2024speed,
    title = {Event Detection from Social Media for Epidemic Prediction},
    author = {Tanmay Parekh and
               Anh Mac and
               Jiarui Yu and
               Yuxuan Dong and
               Syed Shahriar and
               Bonnie Liu and
               Eric Yang and
               Kuan-Hao Huang and 
               Wei Wang and
               Nanyun Peng and
               Kai-Wei Chang},
    booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
    year = {2024},
}

