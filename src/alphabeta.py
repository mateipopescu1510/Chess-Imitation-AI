import pandas as pd
import numpy as np
import torch
from chess import Board, Move
import IPython.display as display
import os
import sys
from encode_data import encode_board, encode_move, decode_move
from model import Net
from train import predict
from stockfish import Stockfish
from timeit import default_timer as timer

stockfish = Stockfish(path='C:\Program Files\stockfish\stockfish-windows-x86-64-avx2.exe',
                      depth=7, parameters={'Threads': 4, 'Hash': 32})


class Node:
    def __init__(self, board: Board, move: Move = None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.value = None

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.board.is_game_over()

    def expand(self, net: Net, net_is_white: bool, n=3, device='cpu'):
        if self.board.turn == net_is_white:
            y_moves, _ = predict(net, self.board, device, n)
            moves = [move for move, _ in y_moves]
        else:
            stockfish.set_fen_position(self.board.fen())
            moves = [Move.from_uci(move['Move'])
                     for move in stockfish.get_top_moves(3)]

        for move in moves:
            new_board = self.board.copy()
            new_board.push(move)
            child = Node(new_board, move, self)
            self.children.append(child)

    def evaluate(self, net: Net, device='cpu'):
        if self.is_terminal():
            result = self.board.result()
            if result == '1-0':
                self.value = 1.0
            elif result == '0-1':
                self.value = -1.0
            else:
                self.value = 0.0
        else:
            _, value = predict(net, self.board, device)
            self.value = value
        return self.value

    def __str__(self):
        return (
            f'NODE fen: {self.board.fen()}\nmove: {str(self.move)} no. children: {len(self.children)} value: {np.around(self.value, 3) if self.value is not None else self.value} is leaf: {self.is_leaf()} is terminal: {self.is_terminal()}')


class Tree:
    def __init__(self, root_board: Board, net: Net, net_is_white=True, max_depth=6, n=3, device='cpu'):
        self.root = Node(root_board)
        self.net = net
        self.net_is_white = net_is_white
        self.max_depth = max_depth
        self.n = n
        self.device = device
        self.transpositions = {}

    def alpha_beta_search(self, node: Node, depth, alpha, beta, maximizing):
        board_hash = hash(node.board.fen())
        if board_hash in self.transpositions:
            # print('TRANSPOSITION')
            entry = self.transpositions[board_hash]
            if entry['depth'] >= depth:
                if entry['type'] == 'exact':
                    return entry['value']
                elif entry['type'] == 'lower_bound':
                    alpha = max(alpha, entry['value'])
                elif entry['type'] == 'upper_bound':
                    beta = min(beta, entry['value'])
                if beta < alpha:
                    return entry['value']

        if depth == 0 or node.is_terminal():
            return node.evaluate(self.net, self.device)

        if node.is_leaf():
            node.expand(self.net, self.net_is_white, self.n, self.device)

        # node.children.sort(key=lambda x: x.evaluate(
        #     self.net, self.device), reverse=maximizing)

        if maximizing:
            value = -1.0
            for child in node.children:
                score = self.alpha_beta_search(
                    child, depth - 1, alpha, beta, False)
                value = max(value, score)
                alpha = max(alpha, score)
                if beta < alpha:
                    break
            # node.value = value
            # return value
        else:
            value = 1.0
            for child in node.children:
                score = self.alpha_beta_search(
                    child, depth - 1, alpha, beta, True)
                value = min(value, score)
                beta = min(beta, score)
                if beta < alpha:
                    break

        node.value = value

        entry_type = 'exact'
        if value < alpha:
            entry_type = 'upper_bound'
        elif value > beta:
            entry_type = 'lower_bound'
        self.transpositions[board_hash] = {
            'value': value, 'depth': depth, 'type': entry_type}

        return value

    def iterative_deepening_dfs(self):
        for depth in range(1, self.max_depth + 1):
            self.alpha_beta_search(
                self.root, depth, -1.0, 1.0, self.net_is_white)

    def best_move(self):
        if self.root.is_terminal():
            return None

        self.alpha_beta_search(
            self.root, self.max_depth, -1.0, 1.0, self.net_is_white)
        # self.iterative_deepening_dfs()

        if self.net_is_white:
            best_child = max(self.root.children, key=lambda x: x.value)
        else:
            best_child = min(self.root.children, key=lambda x: x.value)

        return best_child.move, best_child.value

    def update(self, move):
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.root.parent = None
                return

        new_board = self.root.board.copy()
        new_board.push(move)
        new_root = Node(new_board, move)

        self.root = new_root

    def print_tree(self, node: Node):
        if node.is_leaf() or node.is_terminal():
            return

        for child in node.children:
            print(child)

        for child in node.children:
            self.print_tree(child)


def game_loop(board: Board, net: Net, net_is_white: bool, max_depth=4, n=4, device='cpu'):
    tree = Tree(board, net, net_is_white, max_depth, n, device)
    move = None
    eval = None

    while not tree.root.is_terminal():
        os.system('cls')
        print(f'AI evaluation: {eval}')
        print(tree.root.board)

        if tree.root.board.turn == net_is_white:
            move, eval = tree.best_move()
            tree.update(move)
        else:
            while True:
                move = input('your move: ')
                try:
                    print(move)
                    print(tree.root.board.parse_san(move))
                    move = tree.root.board.parse_san(move)
                    break
                except ValueError:
                    print('invalid move, try again...')
            tree.update(move)

    print('game over!')


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device = 'cpu'
# net = Net().to(device)
# print(net.load_state_dict(torch.load('./models/resd/res20d_300.pth')))

# game_loop(Board(), net, net_is_white=True, max_depth=8, n=3, device=device)
# tree = Tree(Board(), net, net_is_white=True, max_depth=10, n=3, device=device)

# start = timer()
# move, eval = tree.best_move()
# end = timer()

# print(tree.root)
# tree.print_tree(tree.root)
# print(f'best move: {move}; evaluation: {eval:.2f}')
# print(f'finished in {end - start:.2f}s')
