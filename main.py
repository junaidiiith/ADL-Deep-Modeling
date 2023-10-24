from parameters import parse_args
import data_preprocessing
from train_lms import train_gpt2


def main():

    args = parse_args()
    tokenizer = data_preprocessing.get_tokenizer('gpt2')
    train_texts = open('datasets/WeyssowPMC/train/repo_ecore_train.txt').readlines()
    val_texts = open('datasets/WeyssowPMC/train/repo_ecore_val.txt').readlines()

    train_texts = [x.strip() for x in train_texts]
    val_texts = [x.strip() for x in val_texts]

    train_tokenized = data_preprocessing.get_tokenized(train_texts)
    val_tokenized = data_preprocessing.get_tokenized(val_texts)

    train_gpt2(train_tokenized, val_tokenized, tokenizer, args)


main()