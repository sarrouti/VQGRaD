"""Creates a vocabulary using vqgrad_dataset for the vqa dataset.
"""

from collections import Counter
from utils import Vocabulary
#train_
import argparse
import json
import logging
import nltk
import numpy as np
import re
import base64

def process_text(text, vocab, max_length=20):
    """Converts text into a list of tokens surrounded by <start> and <end>.

    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.

    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """
#    if text==None:
#        text="Initial chest radiograph"
    tokens = tokenize(text.lower().strip())
    output = []
    output.append(vocab(vocab.SYM_SOQ))  # <start>
    output.extend([vocab(token) for token in tokens])
    output.append(vocab(vocab.SYM_EOS))  # <end>
    length = min(max_length, len(output))

    return np.array(output[:length]), length


def load_vocab(vocab_path):
    """Load Vocabulary object from a pickle file.

    Args:
        vocab_path: The location of the vocab pickle file.

    Returns:
        A Vocabulary object.
    """
    vocab = Vocabulary()
    vocab.load(vocab_path)
    return vocab


def tokenize(sentence):
    """Tokenizes a sentence into words.

    Args:
        sentence: A string of words.

    Returns:
        A list of words.
    """
    if len(sentence) == 0:
        return []
    if isinstance(sentence,str):
        sentence=sentence
    else:
        sentence=sentence.decode('utf-8')
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    tokens = nltk.tokenize.word_tokenize(
            sentence.strip().lower())
    return tokens


def build_vocab(questions,questions_test,questions_val, cat2ans,cat2caps, threshold):
    """Build a vocabulary from the annotations.

    Args:
        annotations: A json file containing the questions and answers.
        cat2ans: A json file containing answer types.
        threshold: The minimum number of times a work must occur. Otherwise it
            is treated as an `Vocabulary.SYM_UNK`.

    Returns:
        A Vocabulary object.
    """
    with open(questions) as f:
        questions = json.load(f)
    
    with open(cat2ans) as f:
        cat2ans = json.load(f)
    with open(cat2caps) as f:
        cat2caps = json.load(f)
    with open(questions_val) as f:
        questions_val = json.load(f)
    with open(questions_test) as f:
        questions_test = json.load(f)
    words = []
    for category in cat2caps:
        for answer in cat2caps[category]:
            answer = tokenize(answer.encode('utf8'))
            words.extend(answer)
    for category in cat2ans:
        for answer in cat2ans[category]:
            answer = tokenize(answer.encode('utf8'))
            words.extend(answer)

    counter = Counter()
    for i, entry in enumerate(questions):
        question = entry["question"].encode('utf8')
        q_tokens = tokenize(question)
        counter.update(q_tokens)
        if i % 1000 == 0:
            logging.info("Tokenized %d questions." % (i))
    """for j, entry in enumerate(questions_val):
        question = entry["question"].encode('utf8')
        q_tokens = tokenize(question)
        counter.update(q_tokens)
        if j % 1000 == 0:
            logging.info("Tokenized %d questions." % (i))        
    for l, entry in enumerate(questions_test):
        question = entry["question"].encode('utf8')
        q_tokens = tokenize(question)
        counter.update(q_tokens)

        if l % 1000 == 0:
            logging.info("Tokenized %d questions." % (i))"""

    # If a word frequency is less than 'threshold', then the word is discarded.
    words.extend([word for word, cnt in counter.items() if cnt >= threshold])
    words = list(set(words))
    vocab = create_vocab(words)
    return vocab


def create_vocab(words):
    # Adds the words to the vocabulary.
    vocab = Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--questions', type=str,
                        default='data/vqa/'
                        'train_questions_RAD_captions_titles_wiki_imagesAug_v1.json',
                        help='Path for train questions file.')
    parser.add_argument('--questions_val', type=str,
                        default='data/vqa/'
                        'train_questions_RAD_captions_v_wiki_imagesAug_v2.json',
                        help='Path for train questions file.')
    parser.add_argument('--questions_test', type=str,
                        default='data/vqa/'
                        'train_questions_RAD_captions_v_wiki_imagesAug_v2.json',
                        help='Path for train questions file.')
    parser.add_argument('--cat2ans', type=str,
                        default='data/vqa/cat2titles_RAD_wiki_v1.json',
                        help='Path for the answer types.')
    parser.add_argument('--cat2caps', type=str,
                        default='data/vqa/cat2captions_RAD_wiki_v2.json',
                        help='Path for the answer types.')
    # Hyperparameters.
    parser.add_argument('--threshold', type=int, default=1,
                        help='Minimum word count threshold.')

    # Outputs.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_vqgrad.json',
                        help='Path for saving vocabulary wrapper.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab(args.questions,args.questions_test,args.questions_val, args.cat2ans,args.cat2caps, args.threshold)
    logging.info("Total vocabulary size: %d" % len(vocab))
    vocab.save(args.vocab_path)
    logging.info("Saved the vocabulary wrapper to '%s'" % args.vocab_path)
