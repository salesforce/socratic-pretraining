import json
import os
from tqdm import tqdm

dialogue_data_path = '/export/share/datasets/dialogues/pre-training-data/UnDial/processed/undial_v2.json'
dest_data_path = '/export/longdoc-nlp/data/pile-subset-user/dialogue'
if not os.path.exists(dest_data_path):
    os.mkdir(dest_data_path)
dest_data_path = '/export/longdoc-nlp/data/pile-subset-user/dialogue/raw'
if not os.path.exists(dest_data_path):
    os.mkdir(dest_data_path)

with open(dialogue_data_path) as fr:
    data = json.load(fr)

def json_dialogue_to_text(dialogue):
    text = ''
    speakers = {elt['id'] for elt in dialogue['dialog']}
    speakers_dict = {elt:f'Speaker {i}' for i,elt in enumerate(speakers)}
    for elt in dialogue['dialog']:
        text += f'{speakers_dict[elt["id"]]}: {elt["text"]}\n'
    return text

with open(os.path.join(dest_data_path, 'train_undial.jsonl'), 'w') as fw:
    for i, elt in enumerate(tqdm(data[10000:])):
        fw.write(json.dumps({
            'id': i,
            'text': json_dialogue_to_text(elt),
        }) + '\n')

with open(os.path.join(dest_data_path, 'val_undial.jsonl'), 'w') as fw:
    for i, elt in enumerate(tqdm(data[:10000])):
        fw.write(json.dumps({
            'id': i,
            'text': json_dialogue_to_text(elt),
        }) + '\n')