import pandas as pd
import numpy as np
import chess
from timeit import default_timer as timer
import IPython.display as display
import os
import encode_data
from model import Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# boards = torch.from_numpy(np.asarray(
#     np.load('../prepared_data/boards.npy'))).to(device)
# policies = torch.from_numpy(np.asarray(
#     np.load('../prepared_data/policies.npy'))).to(device)
# values = torch.from_numpy(np.asarray(
#     np.load('../prepared_data/values.npy'))).to(device)

# TRAIN_INDEX = 80000

# train_data = TensorDataset(
#     boards[:TRAIN_INDEX], policies[:TRAIN_INDEX], values[:TRAIN_INDEX])
# test_data = TensorDataset(
#     boards[TRAIN_INDEX:], policies[TRAIN_INDEX:], values[TRAIN_INDEX:])


class ChessLoss(nn.Module):
    def __init__(self):
        super(ChessLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum(
            (-policy * (1e-6 + y_policy.float()).float().log()), 1)
        error = (value_error.view(-1).float() + policy_error).mean()
        return error


'''def train(path, epoch_start=0, epoch_stop=20, res_blocks=20, earliest_possible_stop_epoch=100, batch_size=64, net=None):
    if net is None:
        net = Net(res_blocks).to(device)
    net.train()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    criterion = ChessLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epoch_stop - epoch_start)

    losses_per_epoch = []
    mini_batch_size = 50
    temp_path = path + '0' + '.pth'

    for epoch in range(epoch_start, epoch_stop):
        temp_path = path + str(epoch + 1) + '.pth'

        total_loss = 0.0
        losses_per_batch = []

        for i, data in enumerate(train_loader, 0):
            boards, policies, values = data
            optimizer.zero_grad()

            y_policies, y_values = net(boards)
            loss = criterion(y_values.squeeze(), values, y_policies, policies)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % mini_batch_size == mini_batch_size - 1:
                display.clear_output()
                print(
                    f'training res{res_blocks}... epoch ( {epoch + 1}/{epoch_stop - epoch_start} ) @ [ {i + 1}/{len(train_loader)} ] loss: {total_loss / mini_batch_size:.3f}')
                losses_per_batch.append(total_loss / mini_batch_size)
                total_loss = 0.0

        losses_per_epoch.append(sum(losses_per_batch) / len(losses_per_batch))
        scheduler.step()

        if epoch > earliest_possible_stop_epoch:
            if abs(sum(losses_per_epoch[-3:]) / 3 -
                   sum(losses_per_epoch[-23:-20]) / 3) <= 0.001:
                print(f'early stopping triggered at epoch {epoch + 1}...')
                break

        if epoch % 25 == 24:
            print(
                f'saving current model at epoch {epoch + 1} at path {temp_path}...')
            torch.save(net.state_dict(), temp_path)

    print(f'saving current model at epoch {epoch + 1} at path {temp_path}...')
    torch.save(net.state_dict(), temp_path)
    print(f'finished training')
    return net


def test(net, batch_size=64, top_k=3):
    net.eval()

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    total_policy_loss = 0.0
    total_value_loss = 0.0

    correct_top1 = 0
    correct_topk = 0

    total_samples = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            display.clear_output()
            print(f'testing... [ {i + 1}/{len(test_loader)} ]')

            boards, policies, values = data
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)

            y_policies, y_values = net(boards)

            policy_loss = F.cross_entropy(y_policies, policies)
            total_policy_loss += policy_loss.item()

            value_loss = F.mse_loss(y_values.squeeze(), values)
            total_value_loss += value_loss.item()

            policies_indices = torch.argmax(policies, 1)

            _, top1_pred = torch.max(y_policies, 1)
            correct_top1 += (top1_pred == policies_indices).sum().item()

            topk_pred = torch.topk(y_policies, top_k, dim=1).indices
            correct_topk += (topk_pred ==
                             policies_indices.unsqueeze(1)).sum().item()

            total_samples += policies.size(0)

    avg_policy_loss = total_policy_loss / total_samples
    avg_value_loss = total_value_loss / total_samples
    top1_accuracy = correct_top1 / total_samples
    topk_accuracy = correct_topk / total_samples

    print(f"Policy Loss: {avg_policy_loss:.4f}")
    print(f"Value Loss: {avg_value_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-{top_k} Accuracy: {topk_accuracy:.4f}")

    return avg_policy_loss, avg_value_loss, top1_accuracy, topk_accuracy

'''


def predict(net, board: chess.Board, device='cpu', n=5):
    net.eval()

    enc_board = torch.from_numpy(np.asarray(
        encode_data.encode_board(board))).to(device).unsqueeze(0)

    mask = torch.tensor(encode_data.encode_legal_moves(board)
                        ).to(device).unsqueeze(0)

    with torch.no_grad():
        y_policy, y_value = net(enc_board)

    y_policy = y_policy * mask
    y_policy = y_policy / y_policy.sum(dim=1, keepdim=True)

    act, idx = torch.topk(y_policy, n, dim=1)

    non_zero = act > 0
    act = act[non_zero]
    act = act / act.sum()
    idx = idx[non_zero]

    act = act.tolist()
    idx = idx.tolist()

    moves = [encode_data.decode_move(x, board) for x in idx]

    return list(zip(moves, act)), y_value.item()
