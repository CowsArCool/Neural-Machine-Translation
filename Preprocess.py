import os
import pandas as pd
import spacy
import torch
import numpy as np
import argparse

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

torch.manual_seed(0)
np.random.seed(0)


class Vocabulary:
    def __init__(self, freq_threshold, language_tokenizer):
        self.itos = {0: '<pad>', 1: '<SOS>',
                     2: '<EOS>', 3: '<UNK>'}

        self.stoi = [(value, key) for key, value in self.itos.items()]
        self.freq_threshold = freq_threshold
        self.language_tokenizer = language_tokenizer

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        """Tokenizes the given text.

        Args:
            text ([str]): [singe traing example string to tokenize]

        Returns:pyth
            [str]: [tokenized text]
        """
        return [token.text.lower() for token in self.language_tokenizer.tokenizer(text)]

    def build_vocabulary(self, sentences):
        # builds vocabulary from sentences which is a list of strings

        frequencies = {}
        idx = 4

        for sentence in sentences:
            for word in sentence:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

                if frequencies[word] >= self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word

                    idx += 1

    def int_encode(self, text):
        # single sentence as input
        tokenized = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized
        ]


class FlickrDataset (Dataset):
    def __init__(self, df, src_vocab, trg_vocab):
        # df has two columns, one for src sentences and one for trg
        # language tokenizers must be a list with two spacy tokenizers with the first being the src
        self.df = df

        self.src = df['source']  # entire dataset of src
        self.trg = df['targets']  # entire dataset of trg

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.init_token = '<SOS>'
        self.eos_token = '<EOS>'

    def __len__(self):
        return len(self.df.shape[0])

    def __getitem__(self, index):
        """Get a single item from the dataset.

        Args:
            index ([int]): [index of value of pull]

        Returns:
            [torch.tensor]: [source data at index]
            [torch.tensor]: [target data at index]
        """
        src_sentence = self.src[index]
        trg_sentence = self.trg[index]

        # stoi: string to index
        encoded_src_sentence = [self.src_vocab.stoi['<SOS>']]
        encoded_src_sentence += self.src_vocab.int_encode(src_sentence)
        encoded_src_sentence.append(self.src_vocab.stoi['<EOS>'])

        # stoi: string to index
        encoded_trg_sentence = [self.trg_vocab.stoi['<SOS>']]
        encoded_trg_sentence += self.trg_vocab.int_encode(trg_sentence)
        encoded_trg_sentence.append(self.trg_vocab.stoi['<EOS>'])

        return torch.from_numpy(np.array(encoded_src_sentence)), torch.from_numpy(np.array(encoded_trg_sentence))


class MyCollate:
    def __init__(self, pad_idx):
        """Initialize the Pad object .

        Args:
            pad_idx ([str]): [the pad token]
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        function used by DataLoader to construct batches
        """

        sources = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        # uses torch to pad sequences
        targets = pad_sequence(targets, batch_first=False,
                               padding_value=self.pad_idx)

        return sources, targets


def file_exists(dir_name, file_name):
    """Checks if a file exists

    Args:
        dir_name ([path]): [path to directory]
        file_name ([str]): [name of file in directory]

    Returns:
        [path]: [path to file if it exists else None]
    """
    files = os.listdir(dir_name)
    if file_name in files:
        return os.path.join(dir_name, file_name)
    return None


def read_file(path_name):
    df = pd.read_csv(path_name, names=['source', 'target'])
    return df


def makedir_if_needed(directory):
    """Ensure directory if it doesn t exist .

    Args:
        directory ([path]): [path to create dir at]
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def get_test_data_torch(device=torch.device('cpu')):
    spacy_ger = spacy.load('de_core_news_sm')
    spacy_eng = spacy.load('en_core_web_sm')

    init_token = '<sos>'
    eos_token = '<eos>'

    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in spacy_ger.tokenizer(text)]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_eng.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de,
                init_token=init_token,
                eos_token=eos_token,
                lower=True)

    TRG = Field(tokenize=tokenize_en,
                init_token=init_token,
                eos_token=eos_token,
                lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG))

    return test_data


def load_torch_dataset(device=torch.device('cpu'),
                       batch_size=64,
                       max_size=10000,
                       min_freq=2
                       ):
    """
    Loads the multi30k torch dataset.
    Using this dataset will only support the german to english translation task;
    use the eng tokenizer for src and the ger tokenizer for trg

     Arguments:
        None

     Returns:
        BucketIterator ([pd.DataFrame] [dataframe object containing all data in a singe file])

    """

    spacy_ger = spacy.load('de_core_news_sm')
    spacy_eng = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in spacy_ger.tokenizer(text)]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_eng.tokenizer(text)]

    init_token = '<sos>'
    eos_token = '<eos>'

    SRC = Field(tokenize=tokenize_de,
                init_token=init_token,
                eos_token=eos_token,
                lower=True,
                batch_first=True)

    TRG = Field(tokenize=tokenize_en,
                init_token=init_token,
                eos_token=eos_token,
                lower=True,
                batch_first=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=min_freq, max_size=max_size)
    TRG.build_vocab(train_data, min_freq=min_freq, max_size=max_size)

    trainloader, validloader, testloader = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)

    return trainloader, validloader, testloader, SRC.vocab, TRG.vocab


def load_file_dataset(root_dir):
    """Load all files in a directory

    Args:
        root_dir ([path]): [path to directory to scrape for datasets]

    Returns:
        datasets ([pd.DataFrame]): [dataset of train test and valid concatenated together]
    """
    datasets = []
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            file_path = os.path.join(root, name)

            datasets = pd.concat([read_file(file_path), datasets], names=[
                                 'source', 'target'], ignore_index=True)

    assert datasets.shape[1] == [2]

    return datasets


def get_vocab(df, freq_threshold=5, language_tokenizers=[
        spacy.load('de_core_news_sm'), spacy.load('en_core_web_sm')]):
    src = df['source']  # entire dataset of src
    trg = df['targets']  # entire dataset of trg

    src_vocab = Vocabulary(freq_threshold, language_tokenizers[0])
    trg_vocab = Vocabulary(freq_threshold, language_tokenizers[1])

    src_vocab.build_vocabulary(src.tolist())
    trg_vocab.build_vocabulary(trg.tolist())

    return src_vocab, trg_vocab


def get_loaders(df,
                src_vocab=None, trg_vocab=None,
                batch_size=32,
                train_percent=80,
                num_workers=8,
                shuffle=False,
                device='cpu',
                ):
    """Load data from df and converts it into a BucketIterator.

    Args:
        df ([pd.DataFrame]): [df['source'] == src , df['target'] ==  trg]

    Returns:
        loader ([Dataloader]): [Dataloader generator for training]
    """

    assert (src_vocab is not None), 'source vocab is nonexistent'
    assert (trg_vocab is not None), 'target vocab is nonexistent'

    full_dataset_len = len(df)
    pad_idx = src_vocab.stoi['<pad>']

    train_size = full_dataset_len * train_percent/100
    valid_size = full_dataset_len * (1-(train_percent/100))/2
    test_size = full_dataset_len * (1-(train_percent/100))/2

    train_set = df.iloc[:train_size]
    valid_set = df.iloc[train_size:test_size]
    test_set = df.iloc[test_size:]

    loaders = []
    for dataset in [train_set, valid_set, test_set]:
        FlickrData = FlickrDataset(
            dataset, src_vocab, trg_vocab)

        # BucketIterator is depreciated
        # loader = BucketIterator(FlickrData, batch_size=batch_size,
        #                         sort_within_batch=True, sort_key=len(dataset['source']), device=device)

        loader = DataLoader(
            dataset=FlickrData,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=MyCollate(pad_idx=pad_idx),
        )

        loaders.append(loader)

    return loaders


def main(
    root_dir='torch',
    freq_threshold=5,
    batch_size=32,
    num_workers=8,
    train_percent=80,
    shuffle_data=False,
    device=torch.device('cuda')
):
    """Loads the training data and runs the trained model.

    Returns:
        [type]: [description]
    """
    # load english and german tokenization models
    spacy_ger = spacy.load('de_core_news_sm')
    spacy_eng = spacy.load('en_core_web_sm')
    language_tokenizers = [spacy_eng, spacy_ger]

    # parser = argparse.ArgumentParser(description='Preprocessing data')

    # parser.add_argument('__root_dir', type=str,
    #                     help='Pass in the root dir containing seperated data files')
    # # if multiple files are detected in root dir then it will be assumed that they will be merged

    # parser.add_argument('__freq_threshold', type=int, default=5, nargs='?',
    #                     help='Pass in the lowest value for a word to be included in vocab',
    #                     )

    # parser.add_argument('__save_data', type=bool, nargs="?",
    #                     help='Boolean save data to a file after preprocessing',
    #                     default=False)

    # parser.add_argument('__batch_size', type=int, nargs="?",
    #                     help='Batch size for training',
    #                     default=64)

    # parser.add_argument('__num_workers', type=int, nargs="?",
    #                     help='number of workers to DataLoader',
    #                     default=8)

    # parser.add_argument('__train_percent', type=int, nargs="?",
    #                     help='Percentage of data to allocate to the train set',
    #                     default=80)

    # parser.add_argument('__shuffle_data', type=bool, nargs="?",
    #                     help='Boolean whether to suffle data in the DataLoader',
    #                     const=False)

    # args = parser.parse_args()

    if root_dir == 'torch':
        trainloader, validloader, testloader, src_vocab, trg_vocab = (
            load_torch_dataset(device=device, batch_size=batch_size))
        return trainloader, validloader, testloader, src_vocab, trg_vocab

    df = load_file_dataset(root_dir)

    src_vocab, trg_vocab = get_vocab(
        df, freq_threshold=freq_threshold, language_tokenizers=language_tokenizers)

    trainloader, validloader, testloader = get_loaders(df,
                                                       src_vocab=src_vocab, trg_vocab=trg_vocab,
                                                       batch_size=batch_size,
                                                       train_percent=train_percent,
                                                       num_workers=num_workers,
                                                       shuffle=shuffle_data,
                                                       device=device)

    return trainloader, validloader, testloader, src_vocab, trg_vocab


if __name__ == '__main__':
    trainloader, validloader, testloader = main()

    batch = next(iter(trainloader))

    print(f'Batch source shape: {batch.src.size()}')
    print(f'Batch target shape: {batch.trg.size()}' + '\n')
    print(f'Source data sample: {batch.src[0]}')
    print(f'Target data sample: {batch.trg[0]}')

    # python C:\Users\micha\Desktop\Code\MachineLearning\Seq2Seq\Preprocess.py
