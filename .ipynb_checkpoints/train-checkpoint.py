
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import collections

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler, WeightedRandomSampler)
from tqdm import tqdm, trange

from models.modeling_bert import QuestionAnswering, Config
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer

from models.optim import Lamb
from dataloader.dataloader import RawResult, _is_whitespace, SquadExample, read_squad_examples, convert_examples_to_features
import nsml_evaluate
import neptune


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import json

from apex import amp
import neptune


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_cache_examples(args, tokenizer):
    input_file = args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), '_cached_{}_{}_{}_{}'.format('train',
                                                                                               str(args.max_seq_length),
                                                                                               args.doc_stride, args.train_file_name))
    if os.path.exists(cached_features_file):
        print("Loading features from cached file {}".format(cached_features_file))
        examples, weights = read_squad_examples(input_file=args.train_file, set_type='train')
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at {}".format(input_file))
        examples, weights = read_squad_examples(input_file=args.train_file, set_type='train')
        print(len(examples))
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length = args.max_query_length,
                                                is_training=True)

        
        print("Saving features into cached file {}".format(cached_features_file))
        torch.save(features, cached_features_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)
    return dataset,weights
    
def evaluate(args, model, tokenizer, predict_file, eval_examples, eval_features, validation_label):

    """ Eval """
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_example_index)
    sampler = SequentialSampler(dataset)
    #dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers = args.num_workers)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

    model.eval()
    all_results = []
    all_losses = []
    for input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        start_positions = start_positions.cuda()
        end_positions = end_positions.cuda()

        with torch.no_grad():
            output = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            loss, batch_start_logits, batch_end_logits = output['loss'], output['start'], output['end']

            all_losses.append(loss.item())
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_log_odds.json")
    
    predictions, qid_and_answers = nsml_evaluate.compute_predictions_logits(
        eval_examples,
        eval_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
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
    #print(qid_and_answers)
    #print(len(qid_and_answers))
    exact=0
    f1 = 0
    with open(validation_label, "r", encoding='utf-8') as rf:
        for idx, line in enumerate(rf):
            a = line.strip().split('\t')
            if len(a)==1:
                answer = ''
            else:
                answer = a[1]
            #
            if answer != qid_and_answers[idx][1]:
                t = 3
            else:
                exact+=1
            f1+=nsml_evaluate.compute_f1(answer,qid_and_answers[idx][1])
    rf.close()
    
    results = nsml_evaluate.squad_evaluate(eval_examples, predictions)
    
    return results, np.array(all_losses).mean(), exact/len(qid_and_answers), f1/len(qid_and_answers)

def main(args):

    set_seed(args.seed) 
    
   
    tokenizer = BertTokenizer(args.vocab_file, max_len=args.max_seq_length, do_basic_tokenize=True)
    config = Config.from_json_file(args.config_json)
    model = QuestionAnswering(config)
    
    train_dataset, weights = load_and_cache_examples(args, tokenizer)

    """ eval dataset """

    cached_features_file = os.path.join(os.path.dirname(args.predict_file), '_cached_{}_{}_{}_{}'.format('eval',
                                                                                               str(args.max_seq_length),
                                                                                               args.doc_stride, args.predict_file_name))
    if os.path.exists(cached_features_file):
        print("Loading features from cached file {}".format(cached_features_file))
        eval_examples = read_squad_examples(input_file=args.predict_file, set_type='val')
        eval_features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at {}".format(args.predict_file))
        eval_examples = read_squad_examples(input_file=args.predict_file, set_type='val')
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=True)


        print("Saving features into cached file {}".format(cached_features_file))
        torch.save(eval_features, cached_features_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    #return
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    #train_sampler = WeightedRandomSampler(weights.type('torch.DoubleTensor'), len(weights))
    #train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers = args.num_workers)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, betas=(.9, .999), adam=True)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=t_total*args.warm_up, t_total=t_total)
    
    #checkpoint = torch.load('amp_base.pt')
    
    model.cuda()
    model.bert.load_state_dict(torch.load(args.model_file))
    #model.load_state_dict(torch.load(model_file))
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    try:
        os.mkdir(args.output_dir)
    except:
        print('already exist exp')

    model_checkpoint = 'model_start.bin'
    output_model_file = os.path.join(args.output_dir, model_checkpoint)
    torch.save(model.state_dict(), output_model_file)
    

    # Train!
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    print("  Total optimization steps = {}".format(t_total))

    global_step = 0
    epochs = 0
    model.zero_grad()
    model.train()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Train(XX Epoch) Step(X/X) (loss=X.X)", disable=False)
        for step, batch in enumerate(epoch_iterator):
            #total_step = len(epoch_iterator)*epochs+step

            batch = tuple(t.cuda() for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            output = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            loss = output['loss']

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule\
                
                model.zero_grad()
                optimizer.zero_grad()
                epoch_iterator.set_description(
                    "Train(%d Epoch) Step(%d / %d) (loss=%5.5f)" % (_, global_step, t_total, loss.item()))
                neptune.log_metric('loss',x = global_step, y=loss.item())
            
            if (global_step % args.evaluation_step == 0) and (global_step!=0):
                result, val_loss, test_exact, text_f1 = evaluate(args, model, tokenizer, args.predict_file, eval_examples, eval_features, args.validation_label)
                print(test_exact, text_f1)

                neptune.log_metric('exact', x = global_step, y= result['exact'])
                neptune.log_metric('f1', x = global_step, y= result['f1'])
                neptune.log_metric('HasAns_exact', x = global_step, y= result['HasAns_exact'])
                neptune.log_metric('HasAns_f1', x = global_step, y= result['HasAns_f1'])
                neptune.log_metric('NoAns_exact', x = global_step, y= result['NoAns_exact'])
                neptune.log_metric('NoAns_f1', x = global_step, y= result['NoAns_f1'])
                neptune.log_metric('val_loss', x = global_step, y= val_loss)
                neptune.log_metric('test_exact', x = global_step, y= test_exact)
                neptune.log_metric('text_f1', x = global_step, y= text_f1)

                model_checkpoint = 'model_{}.pt'.format(int(global_step))
                output_model_file = os.path.join(args.output_dir, model_checkpoint)
                
                
                torch.save( {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),'loss': loss.item()}, output_model_file)
                model.train()
            del loss
            #del scale_loss
            del batch
            del input_ids
            del input_mask
            del segment_ids
            del start_positions
            del end_positions

            global_step += 1
            
        epochs += 1

    final_model = 'model_final.bin'
    torch.save(model.state_dict(), final_model)
    neptune.stop()
    print('training and evaluation is finished')


if __name__ == '__main__':
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTI3NjBhYzUtNDVjMS00NDAzLWI2N2ItZTM4YTk0OThkM2QwIn0='
    parser = argparse.ArgumentParser(description='Simple training script for training a rotnet network.')

    #env 
    parser.add_argument('--gpus', help = '')
    parser.add_argument('--num_workers',type = int, help = '')
    parser.add_argument('--seed', type = int, help = '')

    #configs
    parser.add_argument('--config_json', type = str, help = '')  
    parser.add_argument('--model_file', type = str, help='')
    parser.add_argument('--vocab_file', type = str, help = '')

    #exp env setting
    parser.add_argument('--num_train_epochs', type = int, help = '')
    parser.add_argument('--train_batch_size', type = int, help = '')
    parser.add_argument('--eval_batch_size', type = int, help = '')
    parser.add_argument('--evaluation_step', type = int, help = '')
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1, help = '')

    #exp hyperparameters
    parser.add_argument('--learning_rate', type = float, help = '')
    parser.add_argument('--warm_up', type = float, help = '')
    parser.add_argument('--adam_epsilon', type = float, help = '')   
    parser.add_argument('--weight_decay', type = float, help = '')
    parser.add_argument('--max_grad_norm', type = float, help = '')
    parser.add_argument('--n_best_size', type = int, help = '')

    #names
    parser.add_argument('--exp_name', type = str, help = '')
    parser.add_argument('--output_dir',type = str,  help = '')
    parser.add_argument('--log_dir',type = str,  help = '')
    parser.add_argument('--train_file',type = str,  help = '')
    parser.add_argument('--predict_file', type = str, help = '')
    parser.add_argument('--train_file_name', type = str, help = '')
    parser.add_argument('--predict_file_name', type = str, help = '')
    parser.add_argument('--validation_label', type = str, help = '')

    #dataset params
    parser.add_argument('--use_korquad', action="store_true", help = '')
    parser.add_argument('--max_seq_length', type = int, help = '')
    parser.add_argument('--doc_stride', type = int, help = '')
    parser.add_argument('--max_query_length', type = int, help = '')
    parser.add_argument('--max_answer_length',type = int, help = '')

    args = parser.parse_args()

    args.output_dir = '{}/{}'.format(args.output_dir, args.exp_name)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    os.environ["NEPTUNE_API_TOKEN"] = api_token

    args_dic = vars(args)
    args_dic['pwd'] = os.getcwd()

    neptune.init('kboseong/nsmlNLP')
    neptune.create_experiment(name = args.exp_name, params = args_dic)

    main(args)
