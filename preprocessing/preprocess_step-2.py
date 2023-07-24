
from numpy import true_divide
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
import argparse
import os
import json
from transformers import set_seed, AutoTokenizer, AutoModelForSeq2SeqLM
from time import time
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=1)  # how many nodes (machines) you have
parser.add_argument('--gpus', type=int, default=-1, help='num gpus per node')
parser.add_argument('--nr', type=int, default=0, help='ranking within the nodes')
parser.add_argument('--model_name', default='Salesforce/mixqg-large', help='Which QG model to use')
parser.add_argument('--num_workers', type=int, default=2, help='number of workes in each gpu for dataloader')
parser.add_argument('--batch_size', type=int, default=28, help='batch size per gpu') # 64 too large with large model.
parser.add_argument('--max_length', type=int, default=32, help='maximum length')
parser.add_argument('--min_length', type=int, default=5, help='minimum length')
parser.add_argument('--num_beams', type=int, default=4, help='number of beams in beam search')
parser.add_argument('--num_beam_groups', type=int, default=1, help='number of goups for diverse beam search')
parser.add_argument('--num_return_sequences', type=int, default=1, help='number of sequences returned for each input example')
parser.add_argument('--diversity_penalty', type=float, default=0, help='between 0 and 1 needs num_beam_groups > 1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bf16', action='store_true', help='whether to use bf16 during inference')
parser.add_argument('--input_dir', default='/data/pile-subset/preprocess_step-1')
parser.add_argument('--input_file', default=None, help='if specified restricts to this file.')
parser.add_argument('--output_dir', default='/data/pile-subset/preprocess_step-2')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--nb_docs", default=None, type=int)

args = parser.parse_args()

def format_inputs(context: str, answer: str):
    return f"{answer} \\n {context}"

def load_jsonl_file(path, nb_docs=None):
    with open(path) as infile:
        data = []
        for i, line in enumerate(infile):
            if nb_docs is not None and i >= nb_docs:
                break
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as out_file:
        for elt in data:
            out_file.write(json.dumps(elt) + '\n')

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
def collate(batch):
        new_batch = {} 
        # new_batch['text'] = [elt['text'] for elt in batch]
        new_batch['sent_id'] = torch.tensor([elt['sent_id'] for elt in batch])
        new_batch['ex_id'] = torch.tensor([elt['ex_id'] for elt in batch])
        new_batch['input_ids'] = tokenizer(
            [elt['text'] for elt in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).input_ids
        return new_batch

def all_gather(q, ws):
    all_qs = [torch.zeros_like(q, dtype=q.dtype).cuda() for _ in range(ws)] 
    dist.all_gather(all_qs, q)
    all_qs = torch.cat(all_qs)
    return all_qs

def all_gather_object(q, ws):
    all_qs = [None for _ in range(ws)]
    dist.all_gather_object(all_qs, q)
    return [elt for l in all_qs for elt in l]

def extract_masked_sentences(batch, idx):
    new_batch = {'ex_id': [], 'sent_id': [], 'text': []}
    for i, ex_id in enumerate(idx):
        document_sents = [s['text'] for s in batch['data'][i]]
        document = ' '.join(document_sents)
        for sent_id, sent in enumerate(document_sents):
            if sent_id in batch['mask_indices'][i]:
                formatted_sentence = format_inputs(document, sent)
                new_batch['ex_id'].append(ex_id)
                new_batch['sent_id'].append(sent_id)
                new_batch['text'].append(formatted_sentence)
    return new_batch

def extract_sentences_to_file(example, ex_id, fw):
    document_sents = [s['text'] for s in example['data']]
    document = ' '.join(document_sents)
    for sent_id, sent in enumerate(document_sents):
        if sent_id in example['mask_indices']:
            formatted_sentence = format_inputs(document, sent)
            out_example = {
                'ex_id': ex_id,
                'sent_id': sent_id,
                'text': formatted_sentence
            }
            fw.write(json.dumps(out_example) + '\n')

def test_model_generation(local_gpu_rank, args):
    set_seed(args.seed)
    args.rank = args.nr * args.gpus + local_gpu_rank  # compute the rank of the current GPU
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        print('Loading model')

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    torch.cuda.set_device(local_gpu_rank)  
    args.device = torch.device("cuda", local_gpu_rank)
    model = model.to(args.device)  # move the model to GPU

    if args.bf16:
        model = model.to(torch.bfloat16)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_gpu_rank], find_unused_parameters=True)
    model.eval()

    # Prepare the dataset
    if args.input_file is None:
        all_files = [f for f in os.listdir(args.input_dir) if f.endswith('.tmp')]  
    else:
        all_files = [f'{args.input_file}.tmp']
    for i_f, f in enumerate(tqdm(all_files)):
        if args.rank == 0:
            print('Preparing dataset:', f)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
                print('Created', args.output_dir)
            
            head = f.replace('.jsonl.tmp', '')
            dest_f = f'{head}_{args.model_name.split("/")[1]}_{args.num_return_sequences}generations.jsonl'
            if os.path.exists(os.path.join(args.output_dir, dest_f)) and not args.overwrite:
                continue
        
        test_dataset = load_dataset('json', data_files={'test':os.path.join(args.input_dir, f)})['test']

        # Prepare dataloader and sampler.
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=test_sampler,
            collate_fn=collate
        )

        if args.rank == 0:
            count = 0
            start_time = time()
            fw = open(os.path.join(args.output_dir, dest_f + '.tmp'), 'w')
            print('Starting generation')
        
        with torch.no_grad():
            for test_batch in test_dataloader:
                for key in test_batch:
                    test_batch[key] = test_batch[key].to(args.device)
                
                outputs = model.module.generate(
                    input_ids=test_batch['input_ids'],
                    max_length=args.max_length,
                    min_length=args.min_length,
                    num_beams=args.num_beams,
                    num_beam_groups=args.num_beam_groups,
                    num_return_sequences=args.num_return_sequences,
                    diversity_penalty=args.diversity_penalty,
                    early_stopping=True,
                )
                
                outputs_sents = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                outputs_with_ids = []
                
                for i, (sent_id, ex_id) in enumerate(zip(test_batch['sent_id'].tolist(),test_batch['ex_id'].tolist())):
                    for ret_id in range(args.num_return_sequences):
                        outputs_with_ids.append({
                            'ex_id': ex_id,
                            'sent_id': sent_id,
                            'question': outputs_sents[i * (args.num_return_sequences) + ret_id],
                        })
            
                outputs_with_ids_pred = all_gather_object(outputs_with_ids, args.world_size)

                if args.rank == 0:
                    for elt in outputs_with_ids_pred:
                        fw.write(json.dumps(elt) + '\n')
                        fw.flush()
                    count += len(outputs_with_ids_pred) // args.num_return_sequences
                    end_time = time()
                    seconds_elapsed = end_time - start_time
                    print(f'Sentences covered {count}/{len(test_dataset)} ({count/len(test_dataset)*100}/100) Time elapsed {seconds_elapsed} ({count/seconds_elapsed} ex/s) Time left {seconds_elapsed/60/60}/{seconds_elapsed/count*len(test_dataset)/60/60}h')

        if args.rank == 0:
            fw.close()
            print(f'Done generating data, starting to aggregate and delete tmp file.')
            data = load_jsonl_file(os.path.join(args.input_dir, f.replace('.tmp', '')), args.nb_docs)
            out_data = load_jsonl_file(os.path.join(args.output_dir, dest_f + '.tmp'))
            print(f'Loaded all outputs')
            # Initialize question field
            for example in data:
                example['questions'] = defaultdict(list)
            for out_example in tqdm(out_data):
                example = data[out_example['ex_id']]
                example['questions'][out_example['sent_id']].append(out_example['question'])
            print(f'Aggregated outputs')
            print(json.dumps(data[0], indent=4))
            save_jsonl(data, os.path.join(args.output_dir, dest_f))
            os.remove(os.path.join(args.output_dir, dest_f + '.tmp'))
            end_time = time()
            seconds_elapsed = end_time - start_time
            print("%s saved! %d/%d finish!" % (dest_f, i_f + 1, len(all_files)))
            print(f'Time elapsed {seconds_elapsed}:')
            print(f'- Sentences covered {count} ({count/seconds_elapsed}/s)')
        
if __name__ == "__main__":
    # Map the doc level data into sentence level data.
    print('Preparing datasets, writing sentences to tmp files.')
    if args.input_file is None:
        all_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]  
    else:
        all_files = [args.input_file]
    print(all_files)
    for f in all_files:
        print('preparing:', f)
        temp_dest = f'{f}.tmp'
        fr = open(os.path.join(args.input_dir, f))
        fw = open(os.path.join(args.input_dir, temp_dest), 'w')
        for ex_id, line in tqdm(enumerate(fr)):
            if args.nb_docs is not None and ex_id >= args.nb_docs:
                break
            example = json.loads(line)
            extract_sentences_to_file(example, ex_id, fw)
        fr.close()
        fw.close()

    # Running the main scrip in parallel
    if args.gpus < 0:
        args.gpus = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='8888'
    mp.spawn(test_model_generation, nprocs=args.gpus, args=(args, ))
    
    # Deleting tmp input files.
    for f in all_files:
        if os.path.exists(os.path.join(args.input_dir, f'{f}.tmp')):
            os.remove(os.path.join(args.input_dir, f'{f}.tmp'))
            print(f'Removing {os.path.join(args.input_dir, f"{f}.tmp")} input data file.')
