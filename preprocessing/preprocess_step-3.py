import torch
import os
import random
import argparse
from multiprocessing import Pool
import json
from tqdm import tqdm
from timeit import default_timer as timer
from itertools import starmap
import random
from collections import Counter
import numpy as np

mask_token = '<mask>'
q_mask_token = '<qmask>'
q_sep_token = '<qsep>'

# Pegasus mode
reconstruct_mode = '<reconstruct>' 
# Pegasus with questions in src
answer_mode = '<answer>'
# Pegasus with question in tgt
ask_and_answer_mode = '<ask&answer>'
# Question generation only
ask_mode = '<ask>'

def load_lines(path, nb_docs=None):
    if nb_docs is not None:
        print(f'Loading {nb_docs} lines.')
    with open(path) as infile:
        if nb_docs is None:
            data = [json.loads(line.strip()) for line in infile.readlines()]
        else:
            data = []
            for i, line in enumerate(infile):
                if i >= nb_docs:
                    break
                data.append(json.loads(line.strip()))
    return data


def load_jsonl_file(input_dir, file_name, nb_docs=None):
    print(f'Loading from {file_name} at {input_dir}')
    if type(file_name) == list:
        data_all = []
        for i_f, f in enumerate(file_name):
            nb = None
            if args.nb_docs is not None:
                if len(args.nb_docs) > 1:
                    nb = int(args.nb_docs[i_f])
                else:
                    nb = int(args.nb_docs[0]) 
            data_all.append(load_lines(os.path.join(input_dir, f), nb))
        if args.nb_docs is None:
            min_len = min([len(d) for d in data_all])
        else:
            min_len = max([len(d) for d in data_all])
        data = []
        for d in data_all:
            data += d[:min_len]
    else:
        data = load_lines(os.path.join(input_dir, file_name))
    print(f'Total lines loaded {len(data)}')
    return data

def save_jsonl(data, path):
    with open(path, 'w') as out_file:
        for elt in data:
            out_file.write(json.dumps(elt) + '\n')

def get_mode_prefix(question, question_in_source, question_instead_of_target):
    if not question:
        return reconstruct_mode
    elif question_in_source:
        return answer_mode
    elif not question_instead_of_target:
        return ask_and_answer_mode
    else:
        return ask_mode

def get_mode(d):
    if d is None:
        return None
    if d['src'].startswith(reconstruct_mode):
        return reconstruct_mode
    elif d['src'].startswith(answer_mode):
        return answer_mode
    elif d['src'].startswith(ask_and_answer_mode):
        return ask_and_answer_mode
    elif d['src'].startswith(ask_mode):
        return ask_mode
    else:
        return None

def remove_mode(src, mode):
    assert mode is None or src.startswith(mode + ' ')
    if mode is not None:
        src = src.replace(mode + ' ', '')
    return src

def separate_questions_text(d):
    q, t, s = None, None, None
    if d is not None:
        if q_sep_token in d['tgt']:
            q = d['tgt'].split(q_sep_token)[0]
            t = d['tgt'].split(q_sep_token)[1]
            s = d['src']
        elif q_sep_token in d['src']:
            q = d['src'].split(q_sep_token)[0]
            t = d['tgt']
            s = d['src'].split(q_sep_token)[1]
        else:
            t = d['tgt']
            s = d['src']
    return q,t,s

def merge_documents(d1, d2):
    """
    Merges two documents
    """
    m1 = get_mode(d1)
    m2 = get_mode(d2)
    assert m1 == m2 or m1 is None or m2 is None, 'merging different modes'
    m = m1 if m1 is not None else m2
    q1, t1, s1 = separate_questions_text(d1)
    q2, t2, s2 = separate_questions_text(d2)

    q = None
    if q1 is not None:
        q = q1
    if q2 is not None and q is None:
        q = q2
    elif q2 is not None and q is not None:
        q = q + '\n' + q2
    
    t = None
    if t1 is not None:
        t = t1
    if t2 is not None and t is None:
        t = t2
    elif t2 is not None and t is not None:
        t = t + '\n' + t2

    s = None
    if s1 is not None:
        s = s1
    if s2 is not None and s is None:
        s = s2
    elif s2 is not None and s is not None:
        s = s + '\n\n' + remove_mode(s2, m2)

    if q is not None:
        # In ask mode the question should be none
        if m == ask_and_answer_mode:
            t = q + q_sep_token + t
        elif m == answer_mode:
            s = q + q_sep_token + s

    cur_docs = []
    if d1 and 'concatenated_docs' in d1:
        cur_docs += d1['concatenated_docs']
    else:
        cur_docs.append(d1)
    if d2 and 'concatenated_docs' in d2:
        cur_docs += d1['concatenated_docs']
    else:
        cur_docs.append(d2)
    merged_doc = {
        'concatenated_docs': cur_docs,
        'src': s,
        'tgt': t,
    }
    return merged_doc

def get_mode_info(info, mode):
    mode_info = info[mode]
    return mode_info['cur_source_len'], mode_info['cur_target_len'], mode_info['cur_doc'], mode_info['cur_count']

def update_mode_info(info, mode, cur_source_len, cur_target_len, cur_doc, cur_count):
    info[mode]['cur_source_len'] = cur_source_len
    info[mode]['cur_target_len'] = cur_target_len
    info[mode]['cur_doc'] = cur_doc
    info[mode]['cur_count'] = cur_count
    return info

def concatenate_examples(documents, counts, args):
    new_documents = []
    new_counts = []
    cur_info = {
        mode: {
            'cur_source_len': 0,
            'cur_target_len': 0,
            'cur_doc': None,
            'cur_count': [],
        }
        for mode in [reconstruct_mode, ask_and_answer_mode, ask_mode, answer_mode]
    }
    print(f'Starting to concatenate {len(documents)} documents')
    for doc, count in tqdm(zip(documents, counts), total=len(documents)):
        mode = get_mode(doc)
        cur_source_len, cur_target_len, cur_doc, cur_count = get_mode_info(cur_info, mode)
        source_len = len(doc['src'].split())
        target_len = len(doc['tgt'].split())
        if source_len > args.max_len_source or target_len > args.max_len_target:
            new_documents.append(doc)
            new_counts.append({'counts': [count], 'source_len': source_len, 'target_len': target_len})
        elif source_len + cur_source_len < args.max_len_source and target_len + cur_target_len < args.max_len_target:
            cur_doc = merge_documents(cur_doc, doc)
            cur_count.append(count)
            cur_source_len += source_len
            cur_target_len += target_len
        elif cur_doc is not None:
            new_documents.append(cur_doc)
            new_counts.append({'counts': cur_count, 'source_len': cur_source_len, 'target_len': cur_target_len})
            cur_doc = doc
            cur_source_len = source_len
            cur_target_len = target_len
            cur_count = [count]
        else:
            cur_doc = doc
            cur_source_len = source_len
            cur_target_len = target_len
            cur_count = [count]
        cur_info = update_mode_info(cur_info, mode, cur_source_len, cur_target_len, cur_doc, cur_count)
    for mode, info in cur_info.items():
        if info['cur_doc'] is not None:
            new_documents.append(info['cur_doc'])
            new_counts.append({
                'counts': info['cur_count'],
                'source_len': info['cur_source_len'],
                'target_len': info['cur_target_len']
            })
    print(f'Concatenated into {len(new_documents)} documents')
    return new_documents, new_counts

def construct_src_tgt(single_doc, i, args):
    mask_indices = single_doc['mask_indices']
    if not 'unchanged_mask_indices' in single_doc:
        non_mask_indices = random.sample(
            list(mask_indices), int(len(mask_indices) * args.non_mask_ratio)
        )
    else:
        non_mask_indices = single_doc['unchanged_mask_indices']


    src, tgt = [single_doc['delimeters'][0]], []
    init_src_questions, init_tgt_questions = [], []
    count = Counter()

    # doc sentences and questions
    sents = [elt['text'] for elt in single_doc['data']]
    questions = None
    if 'questions' in single_doc:
        questions = {int(idx):q for idx,q in single_doc['questions'].items()}

    if questions is None or len(questions) == 0:
        # no questions in the example.
        question = question_in_source = question_instead_of_target = [False] * len(sents)
        count['no_questions'] += 1
    elif args.separate_modes:
        # Simple way to separate the modes is by repeating the random toss for each sentence.
        question = [random.random() < args.question_ratio] * len(sents)
        question_in_source = [random.random() < args.question_src_tgt_ratio] * len(sents)
        question_instead_of_target = [random.random() < args.question_instead_of_tgt_ratio] * len(sents)
    else:
        # Choose mode for each sentence.
        question = (torch.rand(len(sents)) < args.question_ratio).tolist()
        question_in_source = (torch.rand(len(sents)) < args.question_src_tgt_ratio).tolist()
        question_instead_of_target = (torch.rand(len(sents)) < args.question_instead_of_tgt_ratio).tolist()

    if questions is not None and len({q[0] for _,q in questions.items()}) != len(questions):
        # Handle duplicated questions.
        questions_so_far = set()
        for i_s in range(len(sents)):
            if i_s in questions:
                if questions[i_s][0] in questions_so_far:
                    question[i_s] = False
                    question_instead_of_target[i_s] = False
                else:
                    questions_so_far.add(questions[i_s][0])
        count['duplicate_questions'] += 1
    
    for i_s in range(len(sents)):
        if i_s not in mask_indices or len(sents[i_s].split()) < args.min_sent_len:
            src.append(sents[i_s])
            src.append(single_doc['delimeters'][i_s+1])
            if len(sents[i_s].split()) < args.min_sent_len:
                count['short_sentence'] += 1
            count['no_mask'] += 1
        else:
            # questions
            if question[i_s]:
                if question_in_source[i_s]:
                    # src question
                    init_src_questions.append(questions[i_s][0])
                    count['src_question'] += 1
                else:   
                    # tgt questions and qmask in source (should be before src sentence)
                    if not args.separate_modes:
                        # no need of qmasks if we separate by mode (they would be repetitive)
                        src.append(f'{q_mask_token} ')
                        count['src_question_mask'] += 1                        
                    init_tgt_questions.append(questions[i_s][0])
                    count['tgt_question'] += 1

            # src sentence
            # if copy or if we are predicting question as the only target.
            if i_s in non_mask_indices or (question[i_s] and question_instead_of_target[i_s]):
                if question[i_s] and question_instead_of_target[i_s] and args.ask_mask:
                    src.append(f'{mask_token} ')
                    count['src_mask'] += 1
                src.append(sents[i_s])
                count['src_copy'] += 1
            else:
                src.append(f'{mask_token}')
                count['src_mask'] += 1
            src.append(single_doc['delimeters'][i_s+1])
            # tgt sentence
            # add target sentence besides when only predicting question
            if not (question[i_s] and question_instead_of_target[i_s]):
                tgt.append(sents[i_s])
            else:
                count['tgt_question_only'] += 1
                count['src_copy'] -= 1
        count['sent'] += 1

    count['src_question_len_words'] += len(' '.join(init_src_questions).split())
    count['tgt_question_len_words'] += len(' '.join(init_tgt_questions).split())
    count['src_sents_len_words'] += len(''.join(src).split())
    count['tgt_sents_len_words'] += len(' '.join(tgt).split())
    count['tgt_len_words'] = count['tgt_question_len_words'] + count['tgt_sents_len_words']
    count['src_len_words'] = count['src_question_len_words'] + count['src_sents_len_words']

    if len(init_src_questions) > 0:
        init_src_questions.append(q_sep_token)
    if len(init_tgt_questions) > 0 and len(tgt) > 0:
        init_tgt_questions.append(q_sep_token)

    single_doc['src'] = " ".join(init_src_questions) + ' ' + ''.join(src) if len(init_src_questions) > 0 else ''.join(src) 
    single_doc['tgt'] = " ".join(init_tgt_questions + tgt)
    
    # Add mode prefix to source.
    if args.separate_modes:
        mode_prefix = get_mode_prefix(question[0], question_in_source[0], question_instead_of_target[0])
        single_doc['src'] = f'{mode_prefix} ' + single_doc['src']
        count[mode_prefix] += 1

    return single_doc, count

def preprocess(args):
    """
    Generate the pretraining data for wikisum dataset
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('Created', args.output_dir)

    if args.input_file_name is None:
        all_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]  
    else:
        all_files = [args.input_file_name]
    for i_f, f in enumerate(tqdm(all_files)):
        if type(f) == list:
            head = '-'.join([elt.replace('.jsonl', '') for elt in f])
        else:
            head = f.replace('.jsonl', '')
        dest_f = f'{head}_{args.non_mask_ratio}_{args.question_ratio}_{args.question_src_tgt_ratio}_{args.question_instead_of_tgt_ratio}{"_modes" if args.separate_modes else ""}{"_mask" if args.ask_mask else ""}{"_nbdocs"+"-".join(args.nb_docs) if args.nb_docs is not None else ""}.jsonl'
        if os.path.exists(os.path.join(args.output_dir, dest_f)) and not args.overwrite:
            print(f'Skipping {f} as it was already processed. To overwrite use the --overwrite argument.')
            continue
        time_start = timer()
        all_data = load_jsonl_file(args.input_dir, f,  args.nb_docs)
        inputs = [
            (d, i, args)
            for i, d in enumerate(all_data)
        ]
        if args.num_workers > 1:
            processor = Pool(args.num_workers)
            out = processor.starmap(construct_src_tgt, tqdm(inputs, total=len(all_data)))
            processor.close()
        else:
            out = list(starmap(construct_src_tgt, tqdm(inputs, total=len(all_data))))
        new_data, count = [elt[0] for elt in out], [elt[1] for elt in out]
        if args.concatenate_documents_to_max_len:
            new_data, count = concatenate_examples(new_data, count, args)
        print(json.dumps(new_data[0], indent=4))
        if not args.concatenate_documents_to_max_len:
            count = sum(count, Counter())
            stats = {k+'_avg_sent':v/count['sent'] for k,v in count.items() if k != 'sent'}
            stats.update({k+'_avg_example':v/len(new_data) for k,v in count.items()})
        else:
            print(json.dumps(count[0], indent=4))
            stats = {
                'source_len_avg':np.mean([c['source_len'] for c in count]),
                'target_len_avg':np.mean([c['target_len'] for c in count])
            }
        print(json.dumps(stats, indent=4))
        with open(os.path.join(args.output_dir, dest_f + '.stats'), 'w') as out_file:
            out_file.write(json.dumps(stats, indent=4))
        keys_to_keep = ['src', 'tgt']
        final_data = []
        for elt in tqdm(new_data):
            new_elt = {k:v for k,v in elt.items() if k in keys_to_keep}
            final_data.append(new_elt)
        save_jsonl(final_data, os.path.join(args.output_dir, dest_f.replace('.jsonl', '.json')))
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
    parser.add_argument(
        "--non_mask_ratio",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "--question_ratio", 
        default=0.5, 
        type=float, 
        help='How often should we include questions'
    )
    parser.add_argument(
        "--question_src_tgt_ratio",
        default=0.4,
        type=float,
        help='How often should the question be in the src, the rest of the time it will be in the target'
    )
    parser.add_argument(
        "--question_instead_of_tgt_ratio",
        default=0.33,
        type=float,
        help='When the question is in the target, how often to include question instead of target'
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/pile-subset/preprocess_step-3",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/pile-subset/preprocess_step-2",
    )
    parser.add_argument(
        "--overwrite",
        action='store_true'
    )

    parser.add_argument(
        "--num_workers",
        default=1,
        type=int
    )

    parser.add_argument(
        "--separate_modes",
        action='store_true',
        help='if set, separates the types of augmentations per example. So each example only contains one type of augmentation.'
    )

    parser.add_argument(
        "--input_file_name",
        default=None,
        nargs='+',
        help='Only applies to a specific file or files. When multiple files are specified, they are combined into a single output file.'
    )

    parser.add_argument(
        "--nb_docs",
        default=None,
        nargs='+',
        help='if set only processes this number of documents from each file.'
    )

    parser.add_argument(
        "--min_sent_len",
        default=5,
        type=int,
        help='Will not consider sentences shorter than this to be selected as pseudo-summaries'
    )
    parser.add_argument(
        "--ask_mask",
        action='store_true'
    )

    parser.add_argument(
        "--concatenate_documents_to_max_len",
        action='store_true',
        help='Concatenates documents to max len of target or source. Note this expects single mode to be used.'
    )

    parser.add_argument(
        "--max_len_source",
        default=None,
        type=int,
        help='When concatenating documents, the max length of the source. (usually 512)'
    )

    parser.add_argument(
        "--max_len_target",
        default=None,
        type=int,
        help='When concatenating documents, the max length of the target. (usually 256)'
    )


    args = parser.parse_args()
    print(args)

    preprocess(args)
