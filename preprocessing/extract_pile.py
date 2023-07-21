from multiprocessing import Pool
import json
import os
from tqdm import tqdm


# TODO: update these paths.
src_path = '/data/pile'
dest_path = '/data/pile-subset/raw'


files = os.listdir(src_path)

num_workers = 32
file_names = os.listdir(src_path)
file_names = [f for f in file_names if 'jsonl' in f]
pile_set_names = {'Books3'}

def extract_sets(file_name):
    with open(os.path.join(src_path, file_name)) as infile:
        # Open different files for each set
        pile_set_files = {
            pile_set:open(os.path.join(dest_path, f'{file_name.replace(".jsonl", "")}_{pile_set}.jsonl'), 'w') 
            for pile_set in pile_set_names
        }
        for i, line in tqdm(enumerate(infile)):
            j = json.loads(line)
            set_name = j['meta']['pile_set_name']
            if set_name in pile_set_names:
                pile_set_files[set_name].write(line)
                
        # Cleanup
        for f in pile_set_files.values():
            f.close()

with Pool(num_workers) as processor:
    new_data = processor.map(
        extract_sets, file_names
    )
