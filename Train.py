import torch
import torch.optim as optim

import numpy as np
import pathlib
import os

import Predict  # type: ignore

from alive_progress import alive_bar, config_handler


def makedir_if_needed(directory):
    """Ensure directory if it doesn t exist .

    Args:
        directory ([path]): [path to create dir at]
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)


def TrainModel(
    model,
    args,
    trainloader,
    validloader,
    trg_pad_idx,
    src_vocab,
    trg_vocab,
    wandb,
    device
):

    ## Constants ##
    availabe_optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'SGD_nesterov': optim.SGD
    }

    uninitialized_optim = availabe_optimizers.get(args.optimizer)
    if args.optimizer == 'Adam':
        optimizer = uninitialized_optim(model.parameters(),
                                        lr=args.learning_rate)

    elif args.optimizer == 'SGD':
        optimizer = uninitialized_optim(model.parameters(),
                                        lr=args.learning_rate,
                                        momentum=0)

    elif (args.optimizer == 'SGD_nesterov'):
        optimizer = uninitialized_optim(model.parameters(),
                                        lr=args.learning_rate,
                                        momentum=0.5,
                                        nesterov=True)

    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    dataset_length = len(trainloader.dataset)
    number_batches = dataset_length/args.batch_size

    running_dir = pathlib.Path(__file__).parent.absolute()

    # Get directory for all saved models
    saved_models_dir = os.path.join(running_dir, 'saved_models')
    makedir_if_needed(saved_models_dir)

    # Get directory for this specific run
    save_dir = os.path.join(
        saved_models_dir, f'archatecture{model.name}lr{args.learning_rate}batch_size{args.batch_size}\
num_layers{args.num_layers}embedding_size{args.embed_size}optim{args.optimizer}')

    makedir_if_needed(save_dir)
    print(f'[Info] Model Directory: {save_dir}')

    # Runs until another file of the same name at the same location isnt found
    # it increases the count each time allowing for another model
    count = 0
    while (count != -1):
        count += 1
        save_path = os.path.join(save_dir, f'model_{count}')
        if (os.path.isfile(save_path) != True):
            count = -1

    for epoch in range(args.num_epochs):
        model.train()

        cum_loss = []

        with alive_bar(int(np.round(number_batches)),
                       title='Training', bar='smooth',
                       length=75) as bar:
            for batch_num, batch in enumerate(trainloader):
                optimizer.zero_grad()

                src = batch.src.to(device)
                trg = batch.trg.to(device)
                # print(f'Src shape: {src.shape} , Trg shape: {trg.shape}')
                output = model(src, trg[:, :-1])
                # print(
                #     f'Output shape: {output.size()}    Target shape: {trg.shape}')
                output = output.reshape(-1, output.shape[2])
                trg = trg[:, 1:].reshape(-1)
                # print(
                #     f'Output shape: {output.size()}    Target shape: {trg.shape}')

                loss = criterion(output, trg)
                cum_loss.append(loss.item())

                # Runs backpropogation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1)

                optimizer.step()

                wandb.log(
                    {
                        'loss': loss.item(),
                        'epoch': epoch
                    }
                )

                bar.text(f'Epoch Step: {batch_num}')
                bar()

        model.eval()

        test_cum_loss = []
        for test_batch in validloader:

            src = test_batch.src.to(device)
            trg = test_batch.trg.to(device)

            output = model(src, trg[:, :-1])

            output = output.reshape(-1, output.shape[2])
            trg = trg[:, 1:].reshape(-1)

            test_loss = criterion(output, trg)
            test_cum_loss.append(test_loss.item())

        if epoch % args.save_every == 0:
            checkpoint = {
                # saves all epochs in the same file with the epoch in their
                # indexable save name
                f'epoch:{epoch}_state_dict': model.state_dict(),
                f'epoch:{epoch}_optimizer': optimizer.state_dict()
            }

            # Save epoch to file
            torch.save(checkpoint, save_path)

        prediction = Predict.make_prediction(
            model,
            args.test_sentence,
            src_vocab,
            trg_vocab,
            device
        )

        print(f'Prediction: {prediction}')

        wandb.log(
            {'Test Loss': np.mean(test_cum_loss),
             'epoch': epoch
             })

        print(f'\n\033[1mEpoch: {epoch+1} of {args.num_epochs}\033[0m')
        print(f'Loss: {np.round(np.mean(cum_loss), decimals=5)}')
        print(f'Test Loss: {np.round(np.mean(test_cum_loss), decimals=5)}')

    return model
