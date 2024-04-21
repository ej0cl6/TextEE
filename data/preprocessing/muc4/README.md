## MUC-4

MUC-4is a slot filling dataset. We treat it as a event argument extraction task with a dummy trigger at the beginning of the sentence. We further process the data downloaded from [GRIT's repo](https://github.com/xinyadu/grit_doc_event_entity).

### Preprocessing Steps

- Download `train.json`, `dev.json`, and `test.json` from https://github.com/xinyadu/grit_doc_event_entity/tree/master/data/muc/processed and set `MUC4_PATH` in `process_muc4.sh`
- Run `process_muc4.sh`

### Statistics

Split1
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  1020 |    1407    |      1       |       1407       |      5      |    2974    |
| Dev   |  340  |    489     |      1       |        489       |      5      |     918    |
| Test  |  340  |    464     |      1       |        464       |      5      |     884    |

Split2 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  1020 |    1408    |      1       |       1408       |      5      |    2990    |
| Dev   |  340  |    489     |      1       |        489       |      5      |     897    |
| Test  |  340  |    463     |      1       |        463       |      5      |     889    |

Split3
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  1020 |    1419    |      1       |       1419       |      5      |    2912    |
| Dev   |  340  |    473     |      1       |        473       |      5      |     994    |
| Test  |  340  |    468     |      1       |        468       |      5      |     870    |

Split4
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  1020 |    1425    |      1       |       1425       |      5      |    2889    |
| Dev   |  340  |    475     |      1       |        475       |      5      |     921    |
| Test  |  340  |    460     |      1       |        460       |      5      |     966    |

Split5 
|       | #Docs | #Instances | #Event Types | #Events Mentions | #Role Types | #Arguments |
|-------|:-----:|:----------:|:------------:|:----------------:|:-----------:|:----------:|
| Train |  1020 |    1427    |      1       |       1427       |      5      |    2928    |
| Dev   |  340  |    465     |      1       |        465       |      5      |     929    |
| Test  |  340  |    468     |      1       |        468       |      5      |     919    |

### Citation

```bib
@proceedings{muc1992muc4,
  title     = {{F}ourth {M}essage {U}nderstanding {C}onference ({MUC}-4): Proceedings of a Conference Held in {M}c{L}ean, {V}irginia, {J}une 16-18, 1992},
  year      = {1992},
}

