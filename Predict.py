import spacy
import torch
import numpy as np


def make_prediction(model,
                    sentence,
                    src_vocab,
                    trg_vocab,
                    device,
                    max_length=50):
    # print(sentence)

    # Load german tokenizer
    spacy_ger = spacy.load("de_core_news_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')

    # Go through each german token and convert to an index
    text_to_indices = [src_vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    src = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

    outputs = [trg_vocab.stoi["<sos>"]]

    for _ in range(max_length):
        # print(f'outputs shape {len(outputs)}')
        trg = torch.LongTensor(outputs).unsqueeze(0).to(device)

        # print(f'Src: {src.shape} , Trg: {trg.shape}')
        with torch.no_grad():
            output = model(src, trg)

        # print(f'Output: {output.shape}')

        best_guess = output.argmax(2)[:, -1].item()
        # print(
        #     f'Best guess shape: {best_guess.shape}')

        if best_guess == trg_vocab.stoi['<eos>']:
            break

        outputs.append(best_guess)
    translated_sentence = [trg_vocab.itos[idx] for idx in outputs]
    # print(f'Sentence: {translated_sentence}')

    # remove start token
    return translated_sentence[1:]
