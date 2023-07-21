from nltk.tokenize import wordpunct_tokenize
import re

other_speaker_map = {
    'Speaker 0': 'Speaker 1',
    'Speaker 1': 'Speaker 0',
}
token_rules = {
    'Speaker 0': {},
    'Speaker 1': {}
}
replacement_rules = {
    'Speaker 0': {},
    'Speaker 1': {}
}
for current_speaker, replacements in token_rules.items():
    other_speaker = other_speaker_map[current_speaker]
    pronoun_replacements = {
        'me': current_speaker,
        'I': current_speaker,
        'you': other_speaker,
        'your': f"{other_speaker}'s",
        'my': f"{current_speaker}'s",
        'mine': f"{current_speaker}'s",
        'yours': f"{other_speaker}'s",
        'our': f"{current_speaker}'s organization's",
        'us': f"{current_speaker}'s organization"
    }
    cap_replacements = {}
    for k,v in pronoun_replacements.items():
        cap_replacements[k.capitalize()] = v.capitalize()

    replacements.update(pronoun_replacements)
    replacements.update(cap_replacements)

for current_speaker, replacements in replacement_rules.items():
    other_speaker = other_speaker_map[current_speaker]
    verb_replacerment = {
        "'ll": 'will',
        'I am': f'{current_speaker} is',
        "I'm": f'{current_speaker} is',
        "I 'm": f'{current_speaker} is',
        "I m": f'{current_speaker} is',
        'you are': f'{other_speaker} is',
        "you're": f'{other_speaker} is',
        "you 're": f'{other_speaker} is',
        'you re': f'{other_speaker} is',
        'I have': f'{current_speaker} has',
        "I've": f'{current_speaker} has',
        "I 've": f'{current_speaker} has',
        "I ve": f'{current_speaker} has',
        "I do": f'{current_speaker} does',
        "I don't": f"{current_speaker} doesn't",
        "you do": f'{other_speaker} does',
        "you don't": f"{other_speaker} doesn't",
        'you have': f'{other_speaker} has',
        "you've": f'{other_speaker} has',
        "you 've": f'{other_speaker} has',
        'you ve': f'{other_speaker} has',
        'am I': f'is {current_speaker}',
        'are you': f'is {other_speaker}',
        'have I': f'has {current_speaker}',
        'have you': f'has {other_speaker}',
        "do I": f'does {current_speaker}',
        "don't I": f"doesn't {current_speaker}",
        "do you": f'does {other_speaker}',
        "don't you": f"doesn't {other_speaker}",
    }

    cap_replacements = {}
    for k,v in verb_replacerment.items():
        cap_replacements[k.capitalize()] = v.capitalize()

    replacements.update(verb_replacerment)
    replacements.update(cap_replacements)

def apply_rules(rep_rules, tok_rules, sent):
    # apply replacement rules
    for seq, rep in rep_rules.items():
        sent = sent.replace(seq, rep)

    # Tokenize storing whitespaces
    sent_split = wordpunct_tokenize(sent)
    split_string = '|'.join([re.escape(sent) for sent in sent_split])
    delimeters = re.split(split_string, sent)

    # Apply the rule
    for i,tok in enumerate(sent_split):
        if tok in tok_rules:
            sent_split[i] = tok_rules[tok]

    # Restore text with white spaces
    ret_sent = [delimeters[0]]
    for tok, delim in zip(sent_split, delimeters[1:]):
        ret_sent.append(tok)
        ret_sent.append(delim)

    return ''.join(ret_sent)

def turn_dialogue_to_third_person(sentences, delimeters, mask_ids=None):
    current_speaker = None
    modified_text = []
    answers = []
    for i, (delim, sentence) in enumerate(zip(delimeters[:-1], sentences)):
        if 'Speaker 0' in delim:
            current_speaker = 'Speaker 0'
        elif 'Speaker 1' in delim:
            current_speaker = 'Speaker 1'
        elif current_speaker is None:
            return None
        modified_sent = apply_rules(replacement_rules[current_speaker], token_rules[current_speaker], sentence)
        modified_text.append(modified_sent)
        modified_text.append('\n')
        if mask_ids is not None and i in mask_ids:
            answers.append(modified_sent)
    if mask_ids is not None:
        return modified_text, answers
    else:
        return modified_text
