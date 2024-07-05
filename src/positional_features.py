import chess


def material(board, color):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    material = 0

    for piece in piece_values:
        material += len(board.pieces(piece, color)) * piece_values[piece]

    return material


def game_phase(fen):
    board = chess.Board(fen)

    if board.fullmove_number < 16:
        return "opening"

    white_material = material(board, chess.WHITE)
    black_material = material(board, chess.BLACK)

    if white_material < 16 and black_material < 16:
        return "endgame"
    else:
        return "middlegame"


def king_safety_score(board, color):
    return None


def pawn_structure(board, color):
    return None


def piece_activity(board, color):
    return None


def center_control_score(board, color):
    return None


def open_file_control(board, color):
    return None


## etc
