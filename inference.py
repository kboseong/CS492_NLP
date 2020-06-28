import argparse
from models.modeling_bert import QuestionAnswering, Config
import os
import nsml_evaluate
from utils.tokenization import BertTokenizer
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)
from tqdm import tqdm, trange
import logging
import json
import collections
import unicodedata
from utils.korquad_utils import _check_is_max_context, InputFeatures, _improve_answer_span, write_predictions
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle                  
import random
import numpy as np
from nsml_evaluate import compute_f1
from apex import amp
from dataloader.dataloader import RawResult, _is_whitespace, SquadExample, read_squad_examples, convert_examples_to_features

logger = logging.getLogger(__name__)

def read_squad_examples(input_file, set_type):
    is_training = set_type == "train"
    
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    
    examples = []

    has_answer_cnt, no_answer_cnt = 0, 0
    
    for entry in tqdm(input_data[:]):
        qa = entry['qa']
        question_text = qa["question"]
        answer_text = qa['answer']
        if question_text is None or answer_text is None:
            continue

        per_qa_paragraph_cnt = 0
        per_qa_unans_paragraph_cnt = 0
        for pi, paragraph in enumerate(entry["paragraphs"]):
            title = paragraph["title"]
            context_text = str(paragraph["contents"])
            if context_text is None:
                continue
            qas_id = "{}[SEP]{}[SEP]{}".format(question_text, answer_text, pi)
            start_position_character = None
            answers = []

            if answer_text not in context_text:
                is_impossible = True
            else:
                is_impossible = False

            if not is_impossible:
                if is_training:
                    start_position_character = context_text.index(answer_text)  # answer["answer_start"]
                else:
                    answers = [{"text": answer_text,
                                "answer_start": context_text.index(answer_text)}]

            example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=title,
                is_impossible=is_impossible,
                answers=answers,
            )
            if is_impossible:
                #continue
                no_answer_cnt += 1
                per_qa_unans_paragraph_cnt += 1
            else:
                has_answer_cnt += 1

            '''if is_impossible and per_qa_unans_paragraph_cnt > 3:
                continue'''
            # train 메모리때문에 개수제한
            
            '''if is_training and per_qa_paragraph_cnt > 3:
                break'''
            examples.append(example)
            

    print("[{}] Has Answer({}) / No Answer({})".format(set_type, has_answer_cnt, no_answer_cnt))
    return examples


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Simple training script for training a rotnet network.')

    parser.add_argument('--vocab_file', type = str, default = '../pretrained/large_v1/large_v1_32k_vocab.txt')
    parser.add_argument('--predict_file',type = str, default = '../dataset/korquad-open-ldbd/test/test_data/korquad_open_test.json')
    parser.add_argument('--model_config',type = str, default = '../pretrained/large_v1/large_config.json')
    parser.add_argument('--model_weight',type = str, default = '../KorQuad2.0/result/combined_set_aug_2_times_01/model_129000.bin')
    

    args = parser.parse_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    vocab_file = args.vocab_file
    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64
    output_dir = ''
    eval_batch_size = 128
    n_best_size = 20
    max_answer_length = 30
    predict_file_name = args.predict_file.split('/')[-1].split('.')[0]

    tokenizer = BertTokenizer(vocab_file, max_len=max_seq_length, do_basic_tokenize=True)
    config = Config.from_json_file(args.model_config)
    model = QuestionAnswering(config)
    model.load_state_dict(torch.load(args.model_weight)['model_state_dict'])
    model = model.cuda()
    model = amp.initialize(model, opt_level='O1')
    
    cached_features_file = os.path.join(os.path.dirname(args.predict_file), '_cached_{}_{}_{}_{}'.format('test',
                                                                                               str(max_seq_length),
                                                                                               doc_stride, predict_file_name))
    eval_examples = read_squad_examples(input_file=args.predict_file, set_type='val')
    
    if os.path.exists(cached_features_file):
        eval_features = torch.load(cached_features_file)
    else:
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=max_seq_length,
                                                doc_stride=doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=False)
    torch.save(eval_features, cached_features_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_matched_index = torch.tensor([f.example_index for f in eval_features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_matched_index)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)

    logger.info("***** Evaluating *****")
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating!")

    exact = 0 
    index = 0
    for input_ids, input_mask, segment_ids, example_indices, all_matched_index in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            matching_index = all_matched_index[i]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            #tokens = input_ids[i][np.argmax(start_logits):np.argmax(end_logits)+1].detach().cpu().numpy()

            #pred = tokenizer.convert_ids_to_tokens(tokens)
            #answer = tokenizer.tokenize(eval_examples[matching_index].answer_text)


            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
            index+=1

    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(output_dir, "null_log_odds.json")
    
    predictions_best = nsml_evaluate.compute_predictions_logits(
        eval_examples,
        eval_features,
        all_results,
        n_best_size,
        max_answer_length,
        False,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        True,
        True,
        0.0,
        tokenizer,
        is_test=False,
    )
    
    qid_and_answers = [("test-{}".format(qid), answer) for qid, (_, answer) in enumerate(predictions_best[0].items())]
    
    with open(os.path.join(output_dir, 'output'), "w") as outfile:
        outfile.write("\n".join((x[0]+'\t'+x[1] for x in qid_and_answers)))
    outfile.close()