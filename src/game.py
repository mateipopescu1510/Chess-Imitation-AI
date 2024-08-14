import pygame
import torch
import chess
from chess import Board, Move, Square
from alphabeta import Tree
from model import Net
import os

WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
FPS = 15
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOARD_COLOR1 = (240, 217, 181)
BOARD_COLOR2 = (181, 136, 99)

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess AI')

PATH = './pieces/'
PIECE_IMAGES = {}
pieces = ['wp', 'wr', 'wn', 'wb', 'wq', 'wk',
          'bp', 'br', 'bn', 'bb', 'bq', 'bk']

for piece in pieces:
    PIECE_IMAGES[piece] = pygame.transform.scale(
        pygame.image.load(os.path.join(PATH, piece + '.png')),
        (SQUARE_SIZE, SQUARE_SIZE)
    )

'''
def draw_board(board: Board):
    for row in range(8):
        for col in range(8):
            color = BOARD_COLOR1 if (row + col) % 2 == 0 else BOARD_COLOR2
            pygame.draw.rect(WIN, color, pygame.Rect(
                col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for row in range(8):
        for col in range(8):
            piece = board.piece_at(row * 8 + col)
            if piece:
                piece_str = piece.symbol()
                piece_img = PIECE_IMAGES[piece_str.lower(
                ) + ('w' if piece_str.isupper() else 'b')]
                WIN.blit(piece_img, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    pygame.display.update()'''


def draw_board(board: Board):
    WIN.fill(WHITE)

    for row in range(8):
        for col in range(8):
            color = BOARD_COLOR1 if (row + col) % 2 == 0 else BOARD_COLOR2
            pygame.draw.rect(WIN, color, pygame.Rect(
                col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_color = 'w' if piece.color == chess.WHITE else 'b'
            piece_type = piece.symbol().lower()
            piece_img = PIECE_IMAGES[piece_color + piece_type]
            x = chess.square_file(square) * SQUARE_SIZE
            y = (7 - chess.square_rank(square)) * SQUARE_SIZE
            WIN.blit(piece_img, (x, y))

    pygame.display.update()


def get_square_under_mouse(board):
    x, y = pygame.mouse.get_pos()
    row = 7 - (y // SQUARE_SIZE)
    col = x // SQUARE_SIZE
    square = chess.square(col, row)
    return square


def handle_player_move(board: Board, selected):
    square = get_square_under_mouse(board)
    if selected is None:
        if board.piece_at(square) and board.piece_at(square).color == board.turn:
            return square
    else:
        move = Move(selected, square)
        # if (board.piece_at(selected) == chess.PAWN and selected)
        if move in board.legal_moves:
            return move
    return None


def game_loop(board: Board, net: Net, net_is_white: bool, max_depth=8, n=3, device='cpu'):
    tree = Tree(board, net, net_is_white, max_depth, n, device)
    selected = None
    clock = pygame.time.Clock()

    while not tree.root.is_terminal():
        clock.tick(FPS)
        draw_board(tree.root.board)

        if tree.root.board.turn == net_is_white:
            move, eval = tree.best_move()
            tree.update(move)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    move = handle_player_move(tree.root.board, selected)
                    if isinstance(move, chess.Move):
                        tree.update(move)
                        selected = None
                    else:
                        selected = move

        pygame.display.update()

    print('Game over!')
    pygame.quit()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net().to(device)
    net.load_state_dict(torch.load('./models/resd/res20d_300.pth'))
    board = Board()
    game_loop(board, net, net_is_white=True, max_depth=8, n=3, device=device)
