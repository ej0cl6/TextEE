# TextEE

[**Updates**](#updates) | 
[**Datasets**](#supported-datasets) |
[**Models**](#supported-models) |
[**Environment**](#environment) |
[**Running**](#running) |
[**Results**](https://khhuang.me/TextEE/index.html#results) |
[**Website**](https://khhuang.me/TextEE/) |
[**Paper**](https://arxiv.org/abs/2311.09562) 

**Authors**: [Kuan-Hao Huang](https://khhuang.me/), [I-Hung Hsu](https://scholar.google.com/citations?user=OtSSwJgAAAAJ&hl=en), [Tanmay Parekh](https://tanmayparekh.github.io/), [Zhiyu Xie](https://www.linkedin.com/in/zhiyu-xie-607125283/), [Zixuan Zhang](https://zhangzx-uiuc.github.io/), [Premkumar Natarajan](https://www.linkedin.com/in/natarajan/), [Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/), [Nanyun Peng](https://vnpeng.net/), [Heng Ji](https://blender.cs.illinois.edu/hengji.html)

## Introduction

TextEE is a standardized, fair, and reproducible benchmark for evaluating event extraction approaches.
- Standardized data preprocessing for 10+ datasets.
- Standardized data splits for reducing performance variance.
- 10+ implemented event extraction approaches published in recent years.
- Comprehensive reevaluation results for future references. 

Please check mroe details our paper [TextEE: Benchmark, Reevaluation, Reflections, and Future Challenges in Event Extraction](https://arxiv.org/abs/2311.09562). We will keep adding new datasets and new models!

## Updates

- **[04/21/2024]** TextEE supports two more datasets: SPEED and MUC-4.
- **[02/23/2024]** TextEE supports the CEDAR approach now.
- **[12/26/2023]** TextEE supports three more datasets: MLEE, Genia2011, Genia2013.
- **[11/15/2023]** We release TextEE, a framework for reevaluation and benchmark for event extraction. Feel free to contact us (khhuang@illinois.edu) if you want to contribute your models or datasets!

## Supported Datasets

<table  style="width:100%" border="0">
<thead>
<tr class="header">
  <th><strong>Dataset Name</strong></th>
  <th><strong>Task</strong></th>
  <th><strong>Paper Title</strong></th>
  <th><strong>Venue</strong></th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>ACE05</code></td>
  <td> E2E, ED, EAE </td>
  <td> The Automatic Content Extraction (ACE) Program - Tasks, Data, and Evaluation </td>
  <td> LREC 2004 </td>
</tr>
<tr>
  <td><code>ERE</code></td>
  <td> E2E, ED, EAE </td>
  <td> From Light to Rich ERE: Annotation of Entities, Relations, and Events </td>
  <td> EVENTS@NAACL 2015 </td>
</tr>
<tr>
  <td><code>MLEE</code></td>
  <td> E2E, ED, EAE </td>
  <td> Event extraction across multiple levels of biological organization </td>
  <td> Bioinformatics 2012 </td>
</tr>
<tr>
  <td><code>Genia2011</code></td>
  <td> E2E, ED, EAE </td>
  <td> Overview of Genia Event Task in BioNLP Shared Task 2011 </td>
  <td> BioNLP Shared Task 2011 Workshop </td>
</tr>
<tr>
  <td><code>Genia2013</code></td>
  <td> E2E, ED, EAE </td>
  <td> The Genia Event Extraction Shared Task, 2013 Edition - Overview </td>
  <td> BioNLP Shared Task 2013 Workshop </td>
</tr>
<tr>
  <td><code>M2E2</code></td>
  <td> E2E, ED, EAE </td>
  <td> Cross-media Structured Common Space for Multimedia Event Extraction </td>
  <td> ACL 2020 </td>
</tr>
<tr>
  <td><code>CASIE</code></td>
  <td> E2E, ED, EAE </td>
  <td> CASIE: Extracting Cybersecurity Event Information from Text </td>
  <td> AAAI 2020 </td>
</tr>
<tr>
  <td><code>PHEE</code></td>
  <td> E2E, ED, EAE </td>
  <td> PHEE: A Dataset for Pharmacovigilance Event Extraction from Text </td>
  <td> EMNLP 2022 </td>
</tr>
<tr>
  <td><code>MEE</code></td>
  <td> ED </td>
  <td> MEE: A Novel Multilingual Event Extraction Dataset </td>
  <td> EMNLP 2022 </td>
</tr>
<tr>
  <td><code>FewEvent</code></td>
  <td> ED </td>
  <td> Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection </td>
  <td> WSDM 2020 </td>
</tr>
<tr>
  <td><code>MAVEN</code></td>
  <td> ED </td>
  <td> MAVEN: A Massive General Domain Event Detection Dataset </td>
  <td> EMNLP 2020 </td>
</tr>
<tr>
  <td><code>SPPED</code></td>
  <td> ED </td>
  <td> Event Detection from Social Media for Epidemic Prediction </td>
  <td> NAACL 2024 </td>
</tr>
<tr>
  <td><code>MUC-4</code></td>
  <td> EAE </td>
  <td> Fourth Message Understanding Conference </td>
  <td> MUC-4 1992 </td>
</tr>
<tr>
  <td><code>RAMS</code></td>
  <td> EAE </td>
  <td> Multi-Sentence Argument Linking </td>
  <td> ACL 2020 </td>
</tr>
<tr>
  <td><code>WikiEvents</code></td>
  <td> EAE </td>
  <td> Document-Level Event Argument Extraction by Conditional Generation </td>
  <td> NAACL 2021 </td>
</tr>
<tr>
  <td><code>GENEVA</code></td>
  <td> EAE </td>
  <td> GENEVA: Benchmarking Generalizability for Event Argument Extraction with Hundreds of Event Types and Argument Roles </td>
  <td> ACL 2023 </td>
</tr>
</tbody>
</table>

## Supported Models

<table  style="width:100%" border="0">
<thead>
<tr class="header">
  <th><strong>Model Name</strong></th>
  <th><strong>Task</strong></th>
  <th><strong>Paper Title</strong></th>
  <th><strong>Venue</strong></th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>DyGIE++</code></td>
  <td> E2E</td>
  <td> Entity, Relation, and Event Extraction with Contextualized Span Representations </td>
  <td> EMNLP 2019 </td>
</tr>
<tr>
  <td><code>OneIE</code></td>
  <td> E2E </td>
  <td> A Joint Neural Model for Information Extraction with Global Features </td>
  <td> ACL 2020 </td>
</tr>
<tr>
  <td><code>AMR-IE</code></td>
  <td> E2E </td>
  <td> Abstract Meaning Representation Guided Graph Encoding and Decoding for Joint Information Extraction </td>
  <td> NAACL 2021 </td>
</tr>
<tr>
  <td><code>DEGREE</code></td>
  <td> E2E, ED, EAE</td>
  <td> DEGREE: A Data-Efficient Generation-Based Event Extraction Model </td>
  <td> NAACL 2022 </td>
</tr>
<tr>
  <td><code>EEQA</code></td>
  <td> ED, EAE</td>
  <td> Event Extraction by Answering (Almost) Natural Questions </td>
  <td> EMNLP 2020 </td>
</tr>
<tr>
  <td><code>RCEE</code></td>
  <td> ED, EAE</td>
  <td> Event Extraction as Machine Reading Comprehension </td>
  <td> EMNLP 2020 </td>
</tr>
<tr>
  <td><code>Query&Extract</code></td>
  <td> ED, EAE</td>
  <td> Query and Extract: Refining Event Extraction as Type-oriented Binary Decoding </td>
  <td> ACL-Findings 2022 </td>
</tr>
<tr>
  <td><code>TagPrime</code></td>
  <td> ED, EAE</td>
  <td> TAGPRIME: A Unified Framework for Relational Structure Extraction </td>
  <td> ACL 2023 </td>
</tr>
<tr>
  <td><code>UniST</code></td>
  <td> ED</td>
  <td> Unified Semantic Typing with Meaningful Label Inference </td>
  <td> NAACL 2022 </td>
</tr>
<tr>
  <td><code>CEDAR</code></td>
  <td> ED</td>
  <td> GLEN: General-Purpose Event Detection for Thousands of Types </td>
  <td> EMNLP 2023 </td>
</tr>
<tr>
  <td><code>BART-Gen</code></td>
  <td> EAE</td>
  <td> Document-Level Event Argument Extraction by Conditional Generation </td>
  <td> NAACL 2021 </td>
</tr>
<tr>
  <td><code>PAIE</code></td>
  <td> EAE</td>
  <td> Prompt for Extraction? PAIE: Prompting Argument Interaction for Event Argument Extraction </td>
  <td> ACL 2022 </td>
</tr>
<tr>
  <td><code>X-Gear</code></td>
  <td> EAE</td>
  <td> Multilingual Generative Language Models for Zero-Shot Cross-Lingual Event Argument Extraction </td>
  <td> ACL 2022 </td>
</tr>
<tr>
  <td><code>AMPERE</code></td>
  <td> EAE</td>
  <td> AMPERE: AMR-Aware Prefix for Generation-Based Event Argument Extraction Model </td>
  <td> ACL 2023 </td>
</tr>
</tbody>
</table>

## Reevaluation Results

Please check [here](https://khhuang.me/TextEE/index.html#results).

## Environment

1. Please install the following packages from both conda and pip.

```
conda install
  - python 3.8
  - pytorch 2.0.1
  - numpy 1.24.3
  - ipdb 0.13.13
  - tqdm 4.65.0
  - beautifulsoup4 4.11.1
  - lxml 4.9.1
  - jsonlines 3.1.0
  - jsonnet 0.20.0
  - stanza=1.5.0
```
```
pip install
  - transformers 4.30.0
  - sentencepiece 0.1.96
  - scipy 1.5.4
  - spacy 3.1.4
  - nltk 3.8.1
  - tensorboardX 2.6
  - keras-preprocessing 1.1.2
  - keras 2.4.3
  - dgl-cu111 0.6.1
  - amrlib 0.7.1
  - cached_property 1.5.2
  - typing-extensions 4.4.0
  - penman==1.2.2
```
   
  Alternatively, you can use the following command.
```
conda env create -f env.yml
```

2. Run the following command.
```
python -m spacy download en_core_web_lg
```

## Running

### Training
```
./scripts/train.sh [config]
```

### Evaluation for End-to-End Model

```
# Evaluating End-to-End
python TextEE/evaluate_end2end.py --task E2E --data [eval_data] --model [saved_model_folder]

# Evaluating EAE
python TextEE/evaluate_end2end.py --task EAE --data [eval_data] --model [saved_model_folder]
```


### Evaluation for Pipeline Model

```
# Evaluating ED
python TextEE/evaluate_pipeline.py --task ED --data [eval_data] --ed_model [saved_model_folder]

# Evaluating EAE
python TextEE/evaluate_pipeline.py --task EAE --data [eval_data] --eae_model [saved_model_folder]

# Evaluating ED+EAE
python TextEE/evaluate_pipeline.py --task E2E --data [eval_data] --ed_model [saved_model_folder] --eae_model [saved_model_folder]
```

### Making Predictions for New Texts with End-to-End Model

```
# Predicting End-to-End
python TextEE/predict_end2end.py --input_file demo_input.txt --model [saved_model_folder] --output_file demo_output.json
```

### Making Predictions for New Texts with Pipeline Model

```
# Predicting ED+EAE
python TextEE/predict_pipeline.py --input_file demo_input.txt --ed_model [saved_model_folder] --eae_model [saved_model_folder] --output_file demo_output.json
```


## Citation
```bib
@article{Huang23textee,
  author       = {Kuan{-}Hao Huang and
                  I{-}Hung Hsu and
                  Tanmay Parekh and 
                  Zhiyu Xie and
                  Zixuan Zhang and
                  Premkumar Natarajan and
                  Kai{-}Wei Chang and
                  Nanyun Peng and
                  Heng Ji},
  title        = {TextEE: Benchmark, Reevaluation, Reflections, and Future Challenges in Event Extraction},
  journal      = {arXiv preprint arXiv:2311.09562},
  year         = {2023},
}
```
