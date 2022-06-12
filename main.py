# Base packages
import os
import argparse
import wandb

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Internal Modules
import Train  # type: ignore
import Preprocess  # type: ignore
import Predict  # type: ignore
import PearsonModel  # type: ignore

# Models
from TransformerModel import Transformer as Transformer  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparamaters')

    ## Data Loading ##
    parser.add_argument('-root_dir', '-dir',
                        type=str,
                        help='Directory containing data files',
                        required=False,
                        default='torch')

    parser.add_argument('-load_path', '-l_path',
                        type=str,
                        help='Path to a torch model save file',
                        required=False,
                        default=None)

    parser.add_argument('-load_epoch', '-l_epoch',
                        type=int,
                        help='Requires load path to be speicifed, loads the epoch in load_path',
                        required=False,
                        default=None)

    parser.add_argument('-test_sentence', '-sentence',
                        type=str,
                        help='A sentence to translate',
                        required=False,
                        default="Ich habe versucht herauszufinden, wie dumm du bist, aber ich hatte einen Schlaganfall und bin gestorben")
    # fick die franzosen .

    parser.add_argument('-max_len', '-seq_len',
                        type=int,
                        help='Maximum sequence length',
                        required=False,
                        default=100)

    parser.add_argument('-max_vocab_size', '-v_size',
                        type=int,
                        help='Maximum size for vocabularies, any words above this threshold will be deleted',
                        required=False,
                        default=10000)

    parser.add_argument('-min_freq', '-m_freq',
                        type=int,
                        help='Minimum number of times a word must be used for it to be stored as vocab',
                        required=False,
                        default=0)

    ## Model ##
    parser.add_argument('-num_layers', '-n_layers',
                        type=int,
                        help='Number of transformerBlock layers to use in the model',
                        required=False,
                        default=6)

    parser.add_argument('-embed_size', '-e_size',
                        type=int,
                        help='Size of the embedding vectors',
                        required=False,
                        default=512)

    parser.add_argument('-forward_expansion', '-expansion',
                        type=int,
                        help='Rate to upscale the linear layers in TransformerBlock',
                        required=False,
                        default=4)

    parser.add_argument('-attention_heads', '-heads',
                        type=int,
                        help='Number of attention heads to use in the transformer',
                        required=False,
                        default=8)

    parser.add_argument('-dropout_rate', '-dr',
                        type=int,
                        help='Dropout rate to use during training',
                        required=False,
                        default=0.2)

    ## Training ##
    parser.add_argument('-batch_size', '-b_size',
                        type=int,
                        help='Training and testing batch size',
                        required=False,
                        default=64)

    parser.add_argument('-learning_rate', '-lr',
                        type=int,
                        help='Learning rate for optimizer',
                        required=False,
                        default=3e-5)

    parser.add_argument('-num_epochs', '-epochs',
                        type=int,
                        help='Number of training epochs',
                        required=False,
                        default=35)

    parser.add_argument('-optimizer', '-optim',
                        type=str,
                        help='Whether to use gpu for computation input 1 or 0',
                        required=False,
                        default='Adam')

    parser.add_argument('-save_every', '-s_rate',
                        type=int,
                        help='Save the model every (save_every) epochs',
                        required=False,
                        default=5)

    parser.add_argument('-use_cuda', '-gpu',
                        type=int,
                        help='Whether to use gpu for computation input 1 or 0',
                        required=False,
                        default=1)

    args = parser.parse_args()

    return args


def main():
    os.environ["WANDB_SILENT"] = "true"
    # Parses command line arguments, all are optional
    args = parse_args()

    ### Constants ###
    src_pad_token = '<pad>'
    trg_pad_token = '<pad>'

    wandb.init(project='Seq2Seq',
               config=args)
    ### Checks ###

    # PARSE CUDA GPU ACCELERATION ARGUMENT

    if (args.use_cuda == 1):
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    else:
        device = torch.device('cpu')

    print(f'[Info] Using device: {device}')

    trainloader, validloader, testloader, src_vocab, trg_vocab = \
        Preprocess.load_torch_dataset(
            batch_size=args.batch_size,
            max_size=args.max_vocab_size,
            min_freq=args.min_freq,
            device=device
        )
    print('[Info] Dataset Sucessuflly loaded')

    # Explicit variables
    src_pad_idx = src_vocab.stoi[src_pad_token]
    trg_pad_idx = trg_vocab.stoi[trg_pad_token]

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    # # Define Model
    # model = Transformer(
    #     src_vocab_size,
    #     trg_vocab_size,
    #     src_pad_idx,
    #     trg_pad_idx,
    #     args.embed_size,
    #     args.num_layers,
    #     args.forward_expansion,
    #     args.attention_heads,
    #     args.dropout_rate,
    #     device,
    #     args.max_len).to(device)

    model = PearsonModel.Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        device=device).to(device)

    if (args.load_path == None):
        print('[Info] Starting Model Training')
        model = Train.TrainModel(
            model,
            args,
            trainloader,
            validloader,
            trg_pad_idx,
            src_vocab,
            trg_vocab,
            wandb,
            device
        )
    else:
        assert (args.load_path != None and args.load_epoch !=
                None), "Must specify load epoch along with load path"
        model.load_state_dict(torch.load(args.load_path)[
            f'epoch:{args.load_epoch}_state_dict'])

    model.eval()

    prediction = Predict.make_prediction(
        model,
        args.test_sentence,
        src_vocab,
        trg_vocab,
        device
    )
    from pearsonutils import translate_sentence  # type: ignore
    prediction2 = translate_sentence(
        model,
        args.test_sentence,
        src_vocab,
        trg_vocab,
        device
    )

    print(f'Sentence Prediction: {" ".join(prediction)}')

    wandb.finish()


if __name__ == '__main__':
    main()
