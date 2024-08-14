import chess
import numpy as np
import pandas as pd
import positional_features
from stockfish import Stockfish
stockfish = Stockfish(path='C:\Program Files\stockfish\stockfish-windows-x86-64-avx2.exe',
                      depth=10, parameters={'Threads': 4, 'Hash': 32})


def encode_board(data: pd.DataFrame | str | chess.Board):
    if type(data) == str:
        board = chess.Board(data)
    elif type(data) == pd.DataFrame:
        board = chess.Board(data['fen'])
    else:
        board = data

    # board = chess.Board(data) if type(
        # data) == str else chess.Board(data['fen'])

    game_phase = positional_features.game_phase(
        board) if type(data) in [str, chess.Board] else data['game_phase']
    material = positional_features.material_difference(
        board) if type(data) in [str, chess.Board] else data['material']
    king_danger = np.tanh(positional_features.king_danger_score_diff(
        board) / 150) if type(data) in [str, chess.Board] else data['norm_king_danger']
    center_control = np.tanh(positional_features.center_control_score(
        board) / 25) if type(data) in [str, chess.Board] else data['norm_center_control']
    file_control = np.tanh(positional_features.file_control_score(
        board) / 1.5) if type(data) in [str, chess.Board] else data['norm_file_control']
    pawn_structure = positional_features.pawn_structure(
        board) if type(data) in [str, chess.Board] else data['pawn_structure']

    NUM_FEATURES = 24
    pieces = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((8, 8, NUM_FEATURES), dtype=float)
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        tensor[rank, file, pieces[piece.symbol()]] = 1

    tensor[:, :, 12] = board.turn

    tensor[:, :, 13] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[:, :, 14] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[:, :, 15] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[:, :, 16] = board.has_queenside_castling_rights(chess.BLACK)
    tensor[:, :, 17] = board.has_legal_en_passant()

    tensor[:, :, 18] = game_phase
    tensor[:, :, 19] = material
    tensor[:, :, 20] = king_danger
    tensor[:, :, 21] = center_control
    tensor[:, :, 22] = file_control
    tensor[:, :, 23] = pawn_structure

    return tensor


def encode_move(move, board: chess.Board | str):
    if type(board) == str:
        board = chess.Board(board)
    # board = chess.Board(fen)
    encoded = np.zeros((8, 8, 73), dtype=int)

    if type(move) is not chess.Move:
        move = chess.Move.from_uci(move)
    fr = move.from_square
    to = move.to_square
    promotion = move.promotion

    i, j = chess.square_rank(fr), chess.square_file(fr)
    x, y = chess.square_rank(to), chess.square_file(to)
    dx, dy = x - i, y - j

    piece = board.piece_at(fr)

    if piece.symbol() in ["R", "B", "Q", "K", "P", "r", "b", "q", "k", "p"] and promotion in [None, chess.QUEEN]:  # queen moves 0-55
        if dx != 0 and dy == 0:  # north-south idx 0-13
            if dx < 0:
                idx = 7 + dx
            elif dx > 0:
                idx = 6 + dx
        if dx == 0 and dy != 0:  # east-west idx 14-27
            if dy < 0:
                idx = 21 + dy
            elif dy > 0:
                idx = 20 + dy
        if dx == dy:  # NE-SW idx 28-41
            if dx < 0:
                idx = 35 + dx
            if dx > 0:
                idx = 34 + dx
        if dx == -dy:  # NW-SE idx 42-55
            if dx < 0:
                idx = 49 + dx
            if dx > 0:
                idx = 48 + dx
    if piece.symbol() in ["n", "N"]:  # knight moves 56-63
        if (x, y) == (i+2, j-1):
            idx = 56
        elif (x, y) == (i+2, j+1):
            idx = 57
        elif (x, y) == (i+1, j-2):
            idx = 58
        elif (x, y) == (i-1, j-2):
            idx = 59
        elif (x, y) == (i-2, j+1):
            idx = 60
        elif (x, y) == (i-2, j-1):
            idx = 61
        elif (x, y) == (i-1, j+2):
            idx = 62
        elif (x, y) == (i+1, j+2):
            idx = 63
    # underpromotions
    if piece.symbol() in ["p", "P"] and (x == 0 or x == 7) and promotion != None:
        if abs(dx) == 1 and dy == 0:
            if promotion == chess.ROOK:
                idx = 64
            if promotion == chess.KNIGHT:
                idx = 65
            if promotion == chess.BISHOP:
                idx = 66
        if abs(dx) == 1 and dy == -1:
            if promotion == chess.ROOK:
                idx = 67
            if promotion == chess.KNIGHT:
                idx = 68
            if promotion == chess.BISHOP:
                idx = 69
        if abs(dx) == 1 and dy == 1:
            if promotion == chess.ROOK:
                idx = 70
            if promotion == chess.KNIGHT:
                idx = 71
            if promotion == chess.BISHOP:
                idx = 72

    encoded[i, j, idx] = 1
    encoded = encoded.reshape(-1)
    return np.where(encoded == 1)[0][0]


def decode_move(encoded, board: chess.Board):
    tensor = np.zeros(4672, dtype=int)
    tensor[encoded] = 1
    tensor = tensor.reshape((8, 8, 73))

    decoded = None
    promotion = None

    i, j, k = np.where(tensor == 1)
    i, j, k = *i, *j, *k

    if 0 <= k <= 13:
        dy = 0
        if k < 7:
            dx = k - 7
        else:
            dx = k - 6
        to = (7 - (i + dx), j + dy)
    elif 14 <= k <= 27:
        dx = 0
        if k < 21:
            dy = k - 21
        else:
            dy = k - 20
        to = (7 - (i + dx), j + dy)
    elif 28 <= k <= 41:
        if k < 35:
            dy = k - 35
        else:
            dy = k - 34
        dx = dy
        to = (7 - (i + dx), j + dy)
    elif 42 <= k <= 55:
        if k < 49:
            dx = k - 49
        else:
            dx = k - 48
        dy = -dx
        to = (7 - (i + dx), j + dy)
    elif 56 <= k <= 63:
        if k == 56:
            to = (7 - (i + 2), j - 1)
        elif k == 57:
            to = (7 - (i + 2), j + 1)
        elif k == 58:
            to = (7 - (i + 1), j - 2)
        elif k == 59:
            to = (7 - (i - 1), j - 2)
        elif k == 60:
            to = (7 - (i - 2), j + 1)
        elif k == 61:
            to = (7 - (i - 2), j - 1)
        elif k == 62:
            to = (7 - (i - 1), j + 2)
        elif k == 63:
            to = (7 - (i + 1), j + 2)
    else:
        if k == 64:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j)
            else:
                to = (7 - (i + 1), j)
            promotion = chess.ROOK
        if k == 65:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j)
            else:
                to = (7 - (i + 1), j)
            promotion = chess.KNIGHT
        if k == 66:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j)
            else:
                to = (7 - (i + 1), j)
            promotion = chess.BISHOP
        if k == 67:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j - 1)
            else:
                to = (7 - (i + 1), j - 1)
            promotion = chess.ROOK
        if k == 68:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j - 1)
            else:
                to = (7 - (i + 1), j - 1)
            promotion = chess.KNIGHT
        if k == 69:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j - 1)
            else:
                to = (7 - (i + 1), j - 1)
            promotion = chess.BISHOP
        if k == 70:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j + 1)
            else:
                to = (7 - (i + 1), j + 1)
            promotion = chess.ROOK
        if k == 71:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j + 1)
            else:
                to = (7 - (i + 1), j + 1)
            promotion = chess.KNIGHT
        if k == 72:
            if board.turn == chess.BLACK:
                to = (7 - (i - 1), j + 1)
            else:
                to = (7 - (i + 1), j + 1)
            promotion = chess.BISHOP

    if board.piece_at(chess.square(j, 7 - i)) == chess.PAWN and to[0] in [0, 7] and promotion is None:
        promotion = chess.QUEEN

    fr = chess.square(j, i)
    to = chess.square(to[1], 7 - to[0])
    decoded = chess.Move(fr, to, promotion)
    return decoded


def augment_moves(fen, moves_probs):
    move = moves_probs[0][0]

    stockfish.set_fen_position(fen)
    top = stockfish.get_top_moves(10)

    if len(top) == 1:
        return [(top[0]['Move'], 1.0)]

    sf_moves = []
    sf_evals = []
    for m in top:
        sf_moves.append(m['Move'])
        if m['Mate'] is None:
            sf_evals.append(m['Centipawn'])
        else:
            sf_evals.append(m['Mate'] * float('inf'))

    try:
        idx = sf_moves.index(move)
        if idx == len(sf_moves) - 1:
            idx -= 1
    except ValueError:
        idx = len(sf_moves) - 2

    new_moves = [sf_moves[idx], sf_moves[idx + 1]]
    new_probs = [0.5, 0.5]

    if abs(sf_evals[idx] - sf_evals[idx + 1]) > 200:
        new_probs = [0.95, 0.05]

    if idx > 0:
        new_probs.insert(0, 0.5)
        new_probs[:] = [prob / sum(new_probs) for prob in new_probs]

        new_moves.insert(0, sf_moves[idx - 1])

    return list(zip(new_moves, new_probs))


def encode_legal_moves(board: chess.Board | str):
    if type(board) == str:
        board = chess.Board(board)
    # board = chess.Board(fen)
    tensor = np.zeros(4672)
    for move in board.legal_moves:
        encoded = encode_move(move, board)
        tensor[encoded] = 1
    return tensor


def generate_policy(fen, moves_probs):
    policy = np.zeros(4672)

    if len(moves_probs) == 1:
        moves_probs = augment_moves(fen, moves_probs)

    legal_moves = encode_legal_moves(fen)
    policy += legal_moves * 0.001

    for move, prob in moves_probs:
        encoded_move = encode_move(move, fen)
        policy[encoded_move] += prob

    policy /= policy.sum()
    return policy


def evaluate(data: pd.DataFrame | str):
    board = chess.Board(data) if type(
        data) == str else chess.Board(data['fen'])
    game_phase = positional_features.game_phase(
        board) if type(data) == str else data['game_phase']

    material = np.tanh(positional_features.material_difference(
        board) / 2) if type(data) == str else np.tanh(data['material'] / 2)
    king_danger = np.tanh(positional_features.king_danger_score_diff(
        board) / 150) if type(data) == str else data['norm_king_danger']
    center_control = np.tanh(positional_features.center_control_score(
        board) / 25) if type(data) == str else data['norm_center_control']
    file_control = np.tanh(positional_features.file_control_score(
        board) / 1.5) if type(data) == str else data['norm_file_control']
    pawn_structure = positional_features.pawn_structure(
        board) if type(data) == str else data['pawn_structure']

    pawn_structure_scores = {
        0: -1.0,  # open
        1: -0.5,  # semi-open
        2: 0.5,  # semi-closed
        3: 1.0,  # closed
    }
    pawn_score = pawn_structure_scores[pawn_structure]
    if board.turn == chess.BLACK:
        pawn_score *= -1

    if game_phase == 0:  # opening
        material_weight = 0.4
        king_danger_weight = 0.2
        center_control_weight = 0.25
        file_control_score = 0.0
        pawn_score_weight = 0.15
    elif game_phase == 1:  # middlegame
        material_weight = 0.4
        king_danger_weight = 0.25
        center_control_weight = 0.1
        file_control_score = 0.05
        pawn_score_weight = 0.2
    else:  # endgame
        material_weight = 0.7
        king_danger_weight = 0.05
        center_control_weight = 0.05
        file_control_score = 0.2
        pawn_score_weight = 0.0

    eval = (material_weight * material +
            king_danger_weight * king_danger +
            center_control_weight * center_control +
            file_control_score * file_control +
            pawn_score_weight * pawn_score)

    return eval
