"""Transform all the VQG dataset into a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar

from train_utils import Vocabulary
from vocab import load_vocab
from vocab import process_text


def create_answer_mapping(questions_captions, ans2cat,caps2cat):
    """Returns mapping from question_id to answer.

    Only returns those mappings that map to one of the answers in ans2cat.

    Args:
        annotations: VQA annotations file.
        ans2cat: Map from answers to answer categories that we care about.

    Returns:
        answers: Mapping from question ids to answers.
        image_ids: Set of image ids.
    """
    answers = {}
    captions= {}
    image_ids = set()
    for q in questions_captions:
        question_id = q['question_id']
        caption=q['caption']
        answer=q['title']
        if answer in ans2cat:
            answers[question_id] = q['title']            
        if caption in caps2cat:
            captions[question_id] = q['caption']
            image_ids.add(q['image_id'])
#        answer = q['multiple_choice_answer']
#        if answer in ans2cat:
 #           answers[question_id] = answer
  #          image_ids.add(q['image_id'])
    return answers,captions, image_ids


def save_dataset(image_dir, questions, annotations, vocab, ans2cat,caps2cat, output,
                 im_size=224, max_q_length=20, max_a_length=20,
                 with_answers=False):
    """Saves the Visual Genome images and the questions in a hdf5 file.

    Args:
        image_dir: Directory with all the images.
        questions: Location of the questions.
        annotations: Location of all the annotations.
        vocab: Location of the vocab file.
        ans2cat: Mapping from answers to category.
        output: Location of the hdf5 file to save to.
        im_size: Size of image.
        max_q_length: Maximum length of the questions.
        max_a_length: Maximum length of the answers.
        with_answers: Whether to also save the answers.
    """
    # Load the data.
    vocab = load_vocab(vocab)
    with open(annotations) as f:
        annos = json.load(f)
    with open(questions) as f:
        questions = json.load(f)

    # Get the mappings from qid to answers.
    qid2ans,qid2caps, image_ids = create_answer_mapping(annos, ans2cat,caps2cat)
    total_questions = len(qid2ans.keys())
    total_images = len(image_ids)
    print ("Number of images to be written: %d" % total_images)
    print ("Number of QAs to be written: %d" % total_questions)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, im_size, im_size, 3), dtype='f')
    d_answers = h5file.create_dataset(
        "answers", (total_questions, max_a_length), dtype='i')
    d_captions = h5file.create_dataset(
        "captions", (total_questions, max_a_length), dtype='i')
    d_answer_types = h5file.create_dataset(
        "answer_types", (total_questions,), dtype='i')

    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])

    # Iterate and save all the questions and images.
    bar = progressbar.ProgressBar(maxval=total_questions)
    i_index = 0
    q_index = 0
    done_img2idx = {}
    for entry in questions:
        image_id = entry['image_id']
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if question_id not in qid2caps:
            continue
        if image_id not in done_img2idx:
            try:
                path = image_id
                image = Image.open(os.path.join(image_dir, path+".jpg")).convert('RGB')
            except IOError:
                path = image_id
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
            image = transform(image)
            d_images[i_index, :, :, :] = np.array(image)
            done_img2idx[image_id] = i_index
            i_index += 1
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length)
        #print(length)
        d_questions[q_index, :length] = q
        #print(len(d_questions))
        #print(q)
        answer = qid2ans[question_id]
        caption = qid2caps[question_id]
        a, length = process_text(answer, vocab,
                                 max_length=max_a_length)
        c, length2 = process_text(caption, vocab,
                                 max_length=max_a_length)
        d_answers[q_index, :length] = a
        d_captions[q_index, :length2] = c
        d_answer_types[q_index] = int(ans2cat[answer])
        d_indices[q_index] = done_img2idx[image_id]
        q_index += 1
        #print(q_index)
        bar.update(q_index)
    h5file.close()
    print ("Number of images written: %d" % i_index)
    print ("Number of QAs written: %d" % q_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='/train_images_VQ{RAD+Med}',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='data/vqa/'
                        'train_questions_RAD_captions_titles_wiki_imagesAug_v1.json',
                        help='Path for train annotation file.')
    parser.add_argument('--annotations', type=str,
                        default='data/vqa/'
                        'train_questions_RAD_captions_titles_wiki_imagesAug_v1.json',
                        help='Path for train annotation file.')
    parser.add_argument('--cat2ans', type=str,
                        default='data/vqa/cat2titles_RAD_wiki_v1.json',
                        help='Path for the answer types.')
    parser.add_argument('--cat2caps', type=str,
                        default='data/vqa/cat2captions_RAD_wiki_v2.json',
                        help='Path for the answer types.')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_vqgrad.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='data/processed/vqgrad_dataset.hdf5',
                        help='directory for resized images.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=10,
                        help='maximum sequence length for answers.')
    args = parser.parse_args()

    ans2cat = {}
    caps2cat= {}
    with open(args.cat2ans) as f:
        cat2ans = json.load(f)

    cats = sorted(cat2ans.keys())
    with open(args.cat2name, 'w') as f:
        json.dump(cats, f)
    for cat in cat2ans:
        for ans in cat2ans[cat]:
            ans2cat[ans] = cats.index(cat)
    
    with open(args.cat2caps) as f:
        cat2caps = json.load(f)            
    cats = sorted(cat2caps.keys())
    with open(args.cat2name, 'w') as f:
        json.dump(cats, f)
    for cat in cat2caps:
        for ans in cat2caps[cat]:
            caps2cat[ans] = cats.index(cat)            
            
    save_dataset(args.image_dir, args.questions, args.annotations, args.vocab_path,
                 ans2cat,caps2cat, args.output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length)
    print('Wrote dataset to %s' % args.output)
    # Hack to avoid import errors.
    Vocabulary()
