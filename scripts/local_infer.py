# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
import os
import json
from IPython import embed
from tqdm import tqdm

# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

# run interactive local inference
def run(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)

    for data_path in tqdm(args.data_path, desc="Iterating through files"):
        # input text
        results_folder = os.path.basename(os.path.dirname(data_path))
        file_name = os.path.basename(data_path)
        save_path = os.path.join(args.output_path, results_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Load in the data
        data = json.load(open(data_path))
        texts = [e["text"] for e in data]
        probs = []

        for text in tqdm(texts, leave=False, desc="Generating"):
            if len(text) == 0:
                continue

            # evaluate text
            tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                if args.reference_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = reference_model(**tokenized).logits[:, :-1]
                crit = criterion_fn(logits_ref, logits_score, labels)
            # estimate the probability of machine generated text
            # prob = prob_estimator.crit_to_prob(crit)
            # print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
            # print()
            probs.append(crit)
        with open(os.path.join(save_path, file_name), "w") as f:
            json.dump(probs, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-j-6B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="/gscratch/xlab/hallisky/fast-detect-gpt/local_infer_ref")
    parser.add_argument('--output_path', type=str, default="/gscratch/xlab/hallisky/fast-detect-gpt/results")
    parser.add_argument('--data_path', type=str, nargs='+', help='List of data paths separated by spaces')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="/gscratch/xlab/hallisky/cache/")
    args = parser.parse_args()

    run(args)

    """
    python3 local_infer.py --data_path /gscratch/xlab/hallisky/complexity/data/new_book/new_book.json \
        /gscratch/xlab/hallisky/complexity/data/new_book/gpt3_new_book.json \
        /gscratch/xlab/hallisky/complexity/data/new_book/gpt3.5_new_book.json \
        /gscratch/xlab/hallisky/complexity/data/new_book/llama2-70b-chat_new_book_light.json \
        /gscratch/xlab/hallisky/complexity/data/new_book/tulu2-dpo-70b_new_book.json \
        /gscratch/xlab/hallisky/complexity/data/new_book/olmo-7b-internal_new_book.json \
        /gscratch/xlab/hallisky/complexity/data/poems/poems.json \
        /gscratch/xlab/hallisky/complexity/data/poems/gpt3_poems.json \
        /gscratch/xlab/hallisky/complexity/data/poems/gpt3.5_poems.json \
        /gscratch/xlab/hallisky/complexity/data/poems/llama2-70b-chat_poems_prompt-lightest_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/poems/tulu2-dpo-70b_poems_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/poems/olmo-7b-instruct_poems_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/speech/speech.json \
        /gscratch/xlab/hallisky/complexity/data/speech/gpt3_speech.json \
        /gscratch/xlab/hallisky/complexity/data/speech/gpt3.5_speech.json \
        /gscratch/xlab/hallisky/complexity/data/speech/llama2-70b-chat_speech_prompt-lightest_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/speech/tulu2-dpo-70b_speech_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/speech/olmo-7b-instruct_speech_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/fake_news/real_news.json \
        /gscratch/xlab/hallisky/complexity/data/fake_news/gpt3_fake_news.json \
        /gscratch/xlab/hallisky/complexity/data/fake_news/gpt3.5_fake_news.json \
        /gscratch/xlab/hallisky/complexity/data/fake_news/llama2-70b-chat_fake_news_prompt-lightest_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/fake_news/tulu2-dpo-70b_fake_news_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/fake_news/olmo-7b-instruct_fake_news_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/theorem/theorem.json \
        /gscratch/xlab/hallisky/complexity/data/theorem/gpt3_theorem.json \
        /gscratch/xlab/hallisky/complexity/data/theorem/gpt3.5_theorem.json \
        /gscratch/xlab/hallisky/complexity/data/theorem/llama2-70b-chat_theorem_prompt-lightest_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/theorem/tulu2-dpo-70b_theorem_maxtoken164_minlength128.json \
        /gscratch/xlab/hallisky/complexity/data/theorem/olmo-7b-instruct_theorem_maxtoken164_minlength128.json
    """



