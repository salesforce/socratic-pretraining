"""Report mean rouge scores given a file of reference summaries and one or more files of predicted summaries"""

import argparse
from tqdm import tqdm
import json

import stanza

from summ_eval import rouge_metric

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')

def preprocess(text):
    doc = nlp(text)
    return '\n'.join(
        ' '.join(token.text for token in sentence.tokens)
            for sentence in doc.sentences
    )
def mean_length(outputs):
    return sum([len(output.split()) for output in outputs])/len(outputs)

def extract_questions_and_summary(text, guidance):
    questions, summary = '', ''
    if guidance == "questions-first":
        line_split = text.split('<qsep>')
        summary = line_split[-1].strip()
        questions = line_split[0].strip()
    elif guidance == "questions_first_interleaved":
        raise NotImplementedError
    elif guidance == "summary_first":
        raise NotImplementedError
    elif guidance == "summary_first_interleaved":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return questions, summary

def report_mean_rouge(ref_path, pred_paths, args):
    metric = rouge_metric.RougeMetric()

    with open(ref_path) as f:
        refs = []
        refs_question = []
        for line in tqdm(f):
            if args.multi_reference:
                if args.guidance is not None:
                    raise NotImplementedError
                refs.append([preprocess(ref) for ref in json.loads(line)])
            elif args.guidance is not None:
                questions, summary = extract_questions_and_summary(line, args.guidance)
                refs.append(preprocess(summary))
                refs_question.append(preprocess(questions))
            else:
                refs.append(preprocess(line.replace('<qsep>', '')))
        print('First ref')
        print('summary:')
        print(refs[0])
        if args.guidance is not None:
            print('questions:')
            print(refs_question[0])

    for i, pred_path in enumerate(pred_paths):
        with open(pred_path) as f:
            preds = []
            preds_question = []
            if args.json_format:
                preds = json.loads(f.read())
                preds = [preprocess(pred) for pred in preds]
            else:
                for line in tqdm(f, total=len(refs)):
                    line = " ".join(line.replace("###", " ").replace("##", "").split())
                    if args.guidance is not None:
                        questions, summary = extract_questions_and_summary(line, args.guidance)
                        preds.append(preprocess(summary))
                        preds_question.append(preprocess(questions))
                    else:
                        preds.append(preprocess(line.replace('<qsep>', '')))
            if args.guidance is None:
                preds = [pred.split('<qsep>')[-1].strip() if '<qsep>' in pred else pred for pred in preds]
            if i == 0:
                print('First pred')
                print('summary:')
                print(preds[0])
                if args.guidance is not None:
                    print('questions:')
                    print(preds_question[0])
            if not args.multi_reference:
                results = metric.evaluate_batch(preds, refs, aggregate=True)
            else:
                results = []
                for pred, ref_list in zip(preds, refs):
                    for ref in ref_list:
                        pred
                        metric.evaluate_batch([pred], [ref], aggregate=True)
            refs_length = mean_length(refs)
            preds_length = mean_length(preds)
            if args.guidance is not None:
                results_question = metric.evaluate_batch(preds_question, refs_question, aggregate=True)
                refs_questions_length = mean_length(refs_question)
                preds_questions_length = mean_length(preds_question)

            results_dict = {
                "TARGETS": ref_path,
                "PREDICTIONS": pred_path,
                "ROUGE-1": results['rouge']['rouge_1_f_score'] * 100,
                "ROUGE-2": results['rouge']['rouge_2_f_score'] * 100,
                "ROUGE-L": results['rouge']['rouge_l_f_score'] * 100,
                "preds len": preds_length,
                "refs len": refs_length,
            }
            if args.guidance is not None:
                results_dict.update({
                "ROUGE-1 question": results_question['rouge']['rouge_1_f_score'] * 100,
                "ROUGE-2 question": results_question['rouge']['rouge_2_f_score'] * 100,
                "ROUGE-L question": results_question['rouge']['rouge_l_f_score'] * 100,
                "q preds len": preds_questions_length,
                "q refs len": refs_questions_length,
            })
            for k in results_dict:
                if type(results_dict[k]) != str:
                    results_dict[k] = round(results_dict[k], 2)

            with open(f'{pred_path}.rouge', 'w') as fw:
                fw.write(json.dumps(results_dict, indent=4))
            print("=====================")
            print(json.dumps(results_dict, indent=4))
            print("=====================")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-path', help='path to file with reference summaries')
    parser.add_argument('--pred-paths', nargs='+', help='paths to prediction files')
    parser.add_argument(
        '--guidance', 
        default=None, 
        choices=[
            "questions-first",
            "questions_first_interleaved",
            "summary_first",
            "summary_first_interleaved",
        ],help='whether use a guidance')
    parser.add_argument(
        '--multi_reference', 
        action='store_true',
        help='computes the maximum rouge across the different references. Expects a list of references on each line'
    )
    parser.add_argument(
        '--json_format',
        action='store_true',
        help='if the files are stored in json format',
    )
    args = parser.parse_args()
    report_mean_rouge(args.ref_path, args.pred_paths, args)
