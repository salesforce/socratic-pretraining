import itertools
import os
from nltk.tokenize import sent_tokenize
import argparse
from multiprocessing import Pool
import numpy as np
import json
from tqdm import tqdm
from timeit import default_timer as timer
from rouge_score import rouge_scorer
import re
import itertools


def compute_scores(doc_sents, scorer, args):
    scores_sent = []
    text = "\n".join(doc_sents)

    for i_s, s in enumerate(doc_sents):
        # if s is too short, i.e. less than 5 words, we directly set scores to 0
        if len(s.split()) < args.min_sent_len:
            scores_sent.append(
                {
                    "text": s,
                    "rouge1-fmeasure": 0,
                    "rouge2-fmeasure": 0,
                    "rougeL-fmeasure": 0,
                    "rouge-mean-fmeasure": 0,
                }
            )
            continue
        ref_sents = text.replace(s, "")
        score = compute_rouge_scores(scorer, [s], [ref_sents])
        scores_sent.append(
            {
                "text": s,
                "rouge1-fmeasure": score["rouge1"][0].fmeasure,
                "rouge2-fmeasure": score["rouge2"][0].fmeasure,
                "rougeL-fmeasure": score["rougeL"][0].fmeasure,
                "rouge-mean-fmeasure": (
                    score["rouge1"][0].fmeasure
                    + score["rouge2"][0].fmeasure
                    + score["rougeL"][0].fmeasure
                ) / 3,
            }
        )

    assert len(scores_sent) == len(doc_sents)
    return scores_sent


def compute_rouge_scores(scorer, predictions, references):
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    assert len(predictions) == len(references)
    all_scores = []
    for pred, ref in zip(predictions, references):
        all_scores.append(scorer.score(target=ref, prediction=pred))
    final_scores = {}
    for score_type in all_scores[0].keys():
        final_scores[score_type] = [
            all_scores[i][score_type] for i in range(len(all_scores))
        ]
    return final_scores



def select_sentences(doc_data, args):
    # We need to make sure we don't select very short sentences that were given score 0.
    mask_sent_num = int(len(doc_data["data"]) * args.mask_ratio)
    non_zero_count = sum([sent['rouge1-fmeasure'] > 0 for sent in doc_data['data']])
    mask_sent_num = min(mask_sent_num, non_zero_count)
    if args.strategy == "greedy_pegasus_score":
        scores_flat = [s[args.metric] for s in doc_data["data"]]
        mask_indices = np.argsort(scores_flat)[::-1][:mask_sent_num].tolist()
    return mask_indices

def check_requirements(cur_sents, cur_words, args):
    """
    verifies that the current truncated document satisfies the requirements:
    - min num sentences
    - min num words
    """
    check = True
    check = check and len(cur_sents) >= args.min_num_sents
    check = check and cur_words >= args.min_num_words
    return check

def segment_truncate_doc(doc_sents, delims, args):
    """
    Approximate segmentation and truncation of the document.
    """
    truncated_sents_and_delimiters = []

    cur_sents = []
    cur_delimeters = [delims[0]]
    cur_words = 0
    if args.choose_length_based_on_target:
        output_len = args.max_length_output
        if args.consider_question_length:
            output_len = args.max_length_output / (1 + args.question_to_sent_ratio)
        target_len = output_len / (1 - args.mask_ratio)
    else:
        target_len = args.max_length_input / (1 - args.mask_ratio)
    for sent, delim in zip(doc_sents, delims[1:]):
        if cur_words > target_len:
            if not args.segment_document:
                # Return if we are not segmenting the documents
                break
            else:
                # Store current document
                if check_requirements(cur_sents, cur_words, args):
                    truncated_sents_and_delimiters.append(
                        {
                            'sents': cur_sents,
                            'delimeters':cur_delimeters
                        }
                    )
                # Reset for the next document
                cur_words = 0
                cur_sents = []
                cur_delimeters = ['']
        cur_len = len(sent.split())
        if cur_len < args.max_sent_len:
            cur_sents.append(sent)
            cur_delimeters.append(delim)
            cur_words += cur_len

    # Add the last document in the segmented setup or the only document in the normal setup.
    if check_requirements(cur_sents, cur_words, args):
        truncated_sents_and_delimiters.append(
            {
                'sents': cur_sents,
                'delimeters':cur_delimeters
            }
        )
    return truncated_sents_and_delimiters

def segment_data(single_doc, i, args):
    # Tokenize sentences
    if not args.dialogue:
        doc_sents = [s for s in sent_tokenize(single_doc) if s.strip() != ""]    
    else:
        single_doc_turns = re.split("Speaker [0-9]+: ", single_doc)
        doc_sents = [s for turn in single_doc_turns for s in sent_tokenize(turn) if s.strip() != ""]    
    
    # Recover the delimeters (note there will be #sents + 1 delims)
    split_string = '|'.join([re.escape(sent) for sent in doc_sents])
    delimeters = re.split(split_string, single_doc)
    
    # Truncate the doc to have approx the right number of words (should help reduce preprox time)
    segmented_and_truncated_docs_with_delimeters = segment_truncate_doc(doc_sents, delimeters, args)
    return segmented_and_truncated_docs_with_delimeters

def compute_all_scores_single_data(single_doc, i, args):
    doc_sents = single_doc['sents']
    data_with_scores = {}
    data_with_scores['delimeters'] = single_doc['delimeters']
    data_with_scores['num_words_trunc'] = len(' '.join(doc_sents).split())

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    data_with_scores['data'] = compute_scores(doc_sents, scorer, args)

    # select summaries
    data_with_scores['mask_indices'] = select_sentences(data_with_scores, args)
    return data_with_scores

def load_jsonl_file(path, nb_docs=None):
    data = []
    with open(path) as infile:
        for i, line in tqdm(enumerate(infile)):
            if nb_docs is not None and i >= nb_docs:
                break
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as out_file:
        for elt in data:
            out_file.write(json.dumps(elt) + '\n')

def preprocess(args):
    """
    Generate the pretraining data for wikisum dataset
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('Created', args.output_dir)

    if args.input_file_name is None:
        all_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')][::-1]  
    else:
        all_files = [args.input_file_name]
    print(f'Found the following files to preprocess: {all_files}')
    for i_f, f in enumerate(tqdm(all_files)):
        print(f'Working on {f}')
        head = f.replace('.jsonl', '')
        dest_f = f'{head}_{args.strategy}_{args.metric}_{args.mask_ratio}_{args.max_length_input}.jsonl'
        if os.path.exists(os.path.join(args.output_dir, dest_f)) and not args.overwrite:
            continue
        time_start = timer()
        all_data = load_jsonl_file(os.path.join(args.input_dir, f), args.nb_docs)
        print(f'Starting to segment and sentence tokenize the data ({len(all_data)} documents).')
        inputs = [
            (d['text'], i, args)
            for i, d in enumerate(all_data)
            if 'text' in d
        ]
        if args.num_workers > 1:
            processor = Pool(args.num_workers)
            segmented_data = processor.starmap(segment_data, tqdm(inputs, total=len(all_data)))
            processor.close()
        else:
            segmented_data = list(itertools.starmap(segment_data, tqdm(inputs, total=len(all_data))))
        # Flatten the list of lists.
        segmented_data = list(itertools.chain.from_iterable(segmented_data))
        print(f'Done segmenting into {len(segmented_data)} segments, starting to identify pseudo-summaries.')
        inputs = [
            (elt, i, args)
            for i, elt in enumerate(segmented_data)
            if 'sents' in elt and 'delimeters' in elt
        ]
        if args.num_workers > 1:
            processor = Pool(args.num_workers)
            new_data = processor.starmap(compute_all_scores_single_data, tqdm(inputs, total=len(all_data)))
            processor.close()
        else:
            new_data = list(itertools.starmap(compute_all_scores_single_data, tqdm(inputs, total=len(all_data))))
        print(json.dumps(new_data[0], indent=4))
        stats = {
            'num_words_trunc_mean': np.mean([elt['num_words_trunc'] for elt in new_data]),
            'num_words_trunc_median': np.median([elt['num_words_trunc'] for elt in new_data])
        }
        print(json.dumps(stats, indent=4))
        with open(os.path.join(args.output_dir, dest_f + '.stats'), 'w') as out_file:
            out_file.write(json.dumps(stats, indent=4))
        save_jsonl(new_data, os.path.join(args.output_dir, dest_f))
        time_end = timer()
        print("%s saved! %d/%d finish!" % (dest_f, i_f + 1, len(all_files)))
        print("finish one file, time is %f" % (time_end - time_start))


if __name__ == "__main__":
    """
    First step.
    Assumes .jsonl files.
    Produces .pt files.
    """
    parser = argparse.ArgumentParser()
    # Gneral
    parser.add_argument("--mask_ratio", default=0.45, type=float)
    parser.add_argument("--max_length_input", default=512, type=int)
    parser.add_argument("--max_length_output", default=256, type=int)

    parser.add_argument(
        "--choose_length_based_on_target",
        action='store_true',
        help='Should we choose the input length based on the target length. Specify the max_length_output if so.'
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/pile-subset/preprocess_step-1",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/pile-subset/",
    )
    parser.add_argument(
        "--input_file_name", 
        default=None,
        help='if specified only preprocesses this file'
    )
    parser.add_argument("--overwrite", action='store_true')

    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument(
        "--min_sent_len",
        default=5,
        type=int,
        help='Will not consider sentences shorter than this to be selected as pseudo-summaries'
    )

    parser.add_argument(
        "--max_sent_len",
        default=50,
        type=int,
        help='Will drop sentences longer than this from the documents.'
    )

    parser.add_argument(
        "--min_num_sents",
        default=6,
        type=int,
        help='The minimum number of sentences in the documents. Documents with fewer examples will be dropped.'
    )

    parser.add_argument(
        "--min_num_words",
        default=200,
        type=int,
        help='The minimum number of sentences in the documents. Documents with fewer examples will be dropped.'
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "greedy_pegasus_score",
        ],
        default="greedy_pegasus_score",
        help="how to select",
        type=str,
    )
    parser.add_argument(
        "--metric",
        choices=[
            "rouge1-fmeasure",
            "rouge2-fmeasure",
            "rougeL-fmeasure",
            "rouge-mean-fmeasure"
        ],
        default="rouge1-fmeasure",
        type=str,
    )
    parser.add_argument("--nb_docs", default=None, type=int)

    parser.add_argument(
        "--segment_document",
        action='store_true',
        help='Whether to segment the document into chunks of rougly the right size. Separate into sentences based on the max len and the masking ratio.'
    )
    
    parser.add_argument(
        "--dialogue",
        action='store_true',
        help='considers the tokens "Speaker 1: " as delimiters.'
    )

    parser.add_argument(
        "--consider_question_length",
        action='store_true',
        help='Whether to consider the question length in the estimation of the length of the source and target text.'
    )

    parser.add_argument(
        "--question_to_sent_ratio",
        type=float,
        default=0.36,
        help='avg question length / avg pseudosummary sent length (coputed on the books corpus).'
    )

    args = parser.parse_args()
    print(args)
    preprocess(args)
