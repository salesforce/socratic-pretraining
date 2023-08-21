# Socratic Pretraining: Question-Driven Pretraining for Controllable Summarization

This repository accompanies the paper: [Socratic Pretraining: Question-Driven Pretraining for Controllable Summarization](https://arxiv.org/abs/2212.10449) by [Artidoro pagnoni](https://artidoro.github.io/), [Alexander R. Fabbri](https://alex-fabbri.github.io/),
[Wojciech Kryściński](https://twitter.com/iam_wkr), [Chien-Sheng Wu](https://jasonwu0731.github.io/).

We present code and instructions for reproducing the paper experiments and running the models against your own datasets.

## Table of contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Preprocessing](#preprocessing-for-socratic-pretraining)
- [Pretraining](#pretraining-with-socratic-objective)
- [Finetuning](#finetuning-the-socratic-pretrained-model)
- [Using our Checkpoints](#using-our-checkpoints)
- [Citation](#citation)
- [License](#license)

## Introduction
In long document controllable summarization, where labeled data is scarce, pretrained models struggle to adapt to the task and effectively respond to user queries. In [our paper](https://arxiv.org/abs/2212.10449), we introduce Socratic pretraining, a question-driven, unsupervised pretraining objective specifically designed to improve controllability in summarization tasks. By training a model to generate and answer relevant questions in a given context, Socratic pretraining enables the model to more effectively adhere to user-provided queries and identify relevant content to be summarized. We demonstrate the effectiveness of this approach through extensive experimentation on two summarization domains, short stories and dialogue, and multiple control strategies: keywords, questions, and factoid QA pairs. Our pretraining method relies only on unlabeled documents and a question generation system and outperforms pre-finetuning approaches that use additional supervised data. Furthermore, our results show that Socratic pretraining cuts task-specific labeled data requirements in half, is more faithful to user-provided queries, and achieves state-of-the-art performance on QMSum and SQuALITY.

## Requirements
Our code was tested with Python 3.10, PyTorch and Huggingface Transformers.  Follow instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch, then install requirements with:
```
pip install -r requirements.txt
```

## Preprocessing for Socratic Pretraining
For Socratic pretraining we split the preprocessing in three steps: 
1. Compute self-rouge and extract pseudo-summaries, segmenting documents if necessary. 
2. Generate questions about pseudo-summaries. 
3. Combine questions, pseudo-summaries, and documents and prepare the examples for the pretraining phase.

For the dialogue domain, we additionally convert to the third person before applying QG phase. For our PEGASUS pretraining experiments we skip phase 2 of question generation.

Example commands:
1. Data: Obtain the [Pile data](https://pile.eleuther.ai/#:~:text=The%20Pile%20is%20a%20825%20GiB%20diverse%2C%20open,800GB%20Dataset%20of%20Diverse%20Text%20for%20Language%20Modeling). 
2. Extract Data: Using the script `preprocessing/extract_pile.py`, extract only the examples of interest to the `pile-subset` directory. In our case, all those from the Books3 corpus. 
3. Run the preprocessing (more variables are described in the scripts):
```
python preprocess_step-1.py --input_dir=pile-subset/raw --output_dir=pile-subset/preprocess_step-1
python preprocess_step-2.py --input_dir=pile-subset/preprocess_step-1 --output_dir=pile-subset/preprocess_step-2
python preprocess_step-3.py --input_dir=pile-subset/preprocess_step-2 --output_dir=pile-subset/preprocess_step-3

```

## Pretraining with Socratic Objective
We provide sample script for pretraining at `pretraining/example_pretraining_script`. We continue pretraining BART-large on the preprocessed data. Before trying full pretraining experiment, we recommend testing the setup works with the overfitting script.

## Finetuning the Socratic Pretrained Model
To finetune Socratic SegEnc, we follow the finetuning best practices from https://github.com/salesforce/query-focused-sum and initialize the model to the Socratic pretrained BART-large. 

1. Preprocessing: Follow the [instructions](https://github.com/salesforce/query-focused-sum/blob/46cb3878ff9f55b963ad94728676189a6d421d60/preprocessing/README.md) to preprocess the data.
2. Training: You can find an example script at `finetuning/example_finetuning_script.sh`. 
3. Evaluation: For QMSum model evaluation, we use the [SummEval](https://github.com/Yale-LILY/SummEval) toolkit. To compute the ROUGE score you have to clone the repo with:
`git clone https://github.com/Yale-LILY/SummEval.git `
And set `PATH_TO_SUMEVAL` to the path to the cloned repo in the finetuning script. You can use the `qmsum_rouge.py` code to for evaluation. For the SQuALITY model evaluation, we use the same ROUGE setup as the SQuALITY paper. We include the `squality_rouge.py` code to replicate the evaluation. Note in particular that in SQuALITY the authors report the maximum ROUGE score across the various reference summaries provided in the dataset. 

## Using our Checkpoints
We release the Socratic SegEnc checkpoints for QMSum and SQuALITY on the Huggingface model hub.

- QMSum model: https://huggingface.co/Salesforce/qmsum-socratic-books-30M
- SQuALITY model: https://huggingface.co/Salesforce/squality-socratic-books-30M

You can use them as follows:

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="Salesforce/qmsum-socratic-books-30M")
query = "How Did Project Manager and User Interface introduce the prototype of the remote control?"
meeting_transcript = """
Project Manager: Yep . Soon as I get this . Okay . This is our last meeting . Um I'll go ahead and go through the minutes from the previous meeting . Uh and then we'll have a , the prototype presentation .  Um then we will um do an evaluation . Uh or we'll see what , what we need to have under the criteria for the evaluation . Then we'll go through the finance and see if we fall within the budget . Um then we'll do the evaluation , and then we can finish up after that with um any changes that we'll need to make , or hopefully everything will fall right in line . Um let's see , minutes from the last meeting . Um we looked at uh the the trends . We had uh the fashion trends that people want a fancy look-and-feel . It was twice as important as anything else . Um they liked fruit and vegetables in the new styles . Um and a spongy feel . So we were talking about trying to incorporate those into our prototype . Um they wanted limited buttons and simplicity . Um then we looked at the uh the method for coming up with our own remote . Um looking at other other devices . Um the iPod , we really liked the look of that . Um we also had uh the kid's remote for a simple idea . Um a two part remote , which was what were were originally looking at . Uh and then um there was talk of spee uh speech recognition um becoming more uh predominant and easier to use . But I think we've still decided not to go with that .  Then we looked at the components um the materials for the case , the different energy sources , the different types of chips , um and made a decision on what we were going to use to make our remote . Um and basically how , what were making for the prototype . So I'm going to leave it at that and let you guys take over .\n
User Interface: The prototype discussion ...                                     
"""
summarizer(f'<ask&answer> {query} <qsep> {meeting_transcript}')
```

## Citation
When referencing this repository, please cite [this paper](https://arxiv.org/abs/2212.10449):
```bibtex
@article{pagnoni2022socratic,
  title={Socratic Pretraining: Question-Driven Pretraining for Controllable Summarization},
  author={Pagnoni, Artidoro and Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and Wu, Chien-Sheng},
  journal={arXiv preprint arXiv:2212.10449},
  year={2022}
}
```


## License

This repository is released under the [Apache-2.0 License](LICENSE.txt).
