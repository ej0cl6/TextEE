## M2E2

The M2E2 dataset comes from ACL 2020 paper [Cross-media Structured Common Space for Multimedia Event Extraction](https://arxiv.org/abs/2005.02472). We only consider the text part of M2E2. M2E2 is an end-to-end event extraction dataset.

### Preprocessing Steps

- Download dataset from [http://blender.cs.illinois.edu/software/m2e2/m2e2_v0.1/m2e2_annotations.zip](http://blender.cs.illinois.edu/software/m2e2/m2e2_v0.1/m2e2_annotations.zip), uncompress the file, and set `M2E2_PATH` in `process_m2e2.sh`
- Run `process_m2e2.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  4211 |    4211    |      8       |       748        |      15     |    1120    |
| Dev   |   901 |     901    |      8       |       183        |      15     |     280    |
| Test  |   901 |     901    |      8       |       174        |      15     |     259    |

Split2
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  4211 |    4211    |      8       |       794        |      15     |    1171    |
| Dev   |   901 |     901    |      8       |       148        |      14     |     232    |
| Test  |   901 |     901    |      8       |       163        |      15     |     256    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  4211 |    4211    |      8       |       760        |      15     |    1138    |
| Dev   |   901 |     901    |      8       |       160        |      15     |     252    |
| Test  |   901 |     901    |      8       |       185        |      15     |     269    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  4211 |    4211    |      8       |       770        |      15     |    1137    |
| Dev   |   901 |     901    |      8       |       178        |      15     |     276    |
| Test  |   901 |     901    |      8       |       157        |      15     |     246    |

Split5
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  4211 |    4211    |      8       |       747        |      15     |    1122    |
| Dev   |   901 |     901    |      8       |       164        |      14     |     258    |
| Test  |   901 |     901    |      8       |       194        |      15     |     279    |

### Citation

```bib
@inproceedings{Li20m2e2,
  author       = {Manling Li and
                  Alireza Zareian and
                  Qi Zeng and
                  Spencer Whitehead and
                  Di Lu and
                  Heng Ji and
                  Shih{-}Fu Chang},
  title        = {Cross-media Structured Common Space for Multimedia Event Extraction},
  booktitle    = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year         = {2020},
}
```
