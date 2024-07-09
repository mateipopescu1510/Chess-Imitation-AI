import chess

delta_king_zone_central = [
    -9,
    -8,
    -7,
    -1,
    0,
    1,
    7,
    8,
    9,
    15,
    16,
    17,
    23,
    24,
    25,
    31,
    32,
    33,
]
delta_king_zone_A8H1 = [-10, -9, -8, -2, -1,
                        0, 6, 7, 8, 14, 15, 16, 23, 24, 31, 32]
delta_king_zone_A1H8 = [-8, -7, -6, 0, 1,
                        2, 8, 9, 10, 16, 17, 18, 24, 25, 32, 33]

pawn_shield_multiplier = {
    0: 1,
    1: 0.9,
    2: 0.75,
    3: 0.5,
    4: 0.4,
    5: 0.25,
    6: 0.1,
    7: 0.05,
    8: 0,
}
attackers_weight = [0, 0.1, 0.5, 0.75, 0.88, 0.94, 0.97, 0.99, 1, 1, 1, 1, 1]
attack_values = {
    chess.PAWN: 10,
    chess.KNIGHT: 20,
    chess.BISHOP: 20,
    chess.ROOK: 40,
    chess.QUEEN: 80,
    chess.KING: 10,
}

center_attack_weight = {
    chess.PAWN: 10,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 2,
    chess.QUEEN: 1,
    chess.KING: 1,
}

rank_multiplier = {2: 0.25, 3: 0.75, 4: 1.0, 5: 2.0}

file_control_piece_multiplier = {chess.ROOK: 1.0, chess.QUEEN: 0.5}

# squares = [
#     56, 57, 58, 59, 60, 61, 62, 63,
#     48, 49, 50, 51, 52, 53, 54, 55,
#     40, 41, 42, 43, 44, 45, 46, 47,
#     32, 33, 34, 35, 36, 37, 38, 39,
#     24, 25, 26, 27, 28, 29, 30, 31,
#     16, 17, 18, 19, 20, 21, 22, 23,
#      8,  9, 10, 11, 12, 13, 14, 15,
#      0,  1,  2,  3,  4,  5,  6,  7
#     ]

big_center = [
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.D4,
    chess.E4,
    chess.F4,
    chess.C5,
    chess.D5,
    chess.E5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
]


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


def game_phase(board):
    if board.fullmove_number < 16:
        return "opening"

    white_material = material(board, chess.WHITE)
    black_material = material(board, chess.BLACK)

    if white_material < 16 and black_material < 16:
        return "endgame"
    else:
        return "middlegame"


def king_proximity_multiplier(distance):
    if distance <= 10:
        return 2
    if 10 < distance < 20:
        return 1
    if 20 < distance < 30:
        return 0.5
    if 30 < distance < 40:
        return 0.1
    return 0


def get_king_zone(board, color):
    direction = 1 if color == chess.WHITE else -1

    king_square = board.king(color)
    king_file = chess.square_file(king_square)

    king_zone = []
    delta_king_zone = []

    if king_file not in [0, 7]:
        delta_king_zone = delta_king_zone_central
    elif color == chess.WHITE:
        if king_file == 0:
            delta_king_zone = delta_king_zone_A1H8
        else:
            delta_king_zone = delta_king_zone_A8H1
    else:
        if king_file == 0:
            delta_king_zone = delta_king_zone_A8H1
        else:
            delta_king_zone = delta_king_zone_A1H8

    for delta in delta_king_zone:
        if 0 < king_square + delta * direction < 64:
            king_zone.append(king_square + delta * direction)

    return king_zone


def king_zone_attackers_score(board, color):
    king_zone = get_king_zone(board, color)
    king_square = board.king(color)

    attackers = set()
    attackersCount = 0
    score = 0
    for square in king_zone:
        distance = abs(square - king_square)
        proximity_multiplier = king_proximity_multiplier(distance)

        for attacker in board.attackers(not color, square):
            attackers.add(attacker)
            attackersCount = len(attackers)
            score += attack_values[board.piece_type_at(
                attacker)] * proximity_multiplier

    score *= attackers_weight[attackersCount]
    return score


def pawn_shield_score(board, color):
    king_square = board.king(color)
    king_pawn_zone = [
        square
        for square in get_king_zone(board, color)
        if abs(square - king_square) < 20
    ]

    pawnsCount = 0
    for square in king_pawn_zone:
        if (
            board.piece_type_at(square) == chess.PAWN
            and board.color_at(square) == color
        ):
            pawnsCount += 1

    score = pawn_shield_multiplier[pawnsCount]
    return score


def king_danger_score(board, user_side):
    user_king_danger = king_zone_attackers_score(
        board, user_side) * pawn_shield_score(board, user_side)
    opponent_king_danger = king_zone_attackers_score(
        board, not user_side) * pawn_shield_score(board, not user_side)
    return (
        round(user_king_danger, 2),
        round(opponent_king_danger, 2),
        round(opponent_king_danger - user_king_danger, 2),
    )


def square_control_score(board, square, color):
    score = 0
    attackers = board.attackers(color, square)
    for attacker in attackers:
        score += center_attack_weight[board.piece_type_at(attacker)]

    if board.piece_type_at(square) == chess.PAWN and board.color_at(square) == color:
        score += 10

    rank = (
        chess.square_rank(square)
        if color == chess.WHITE
        else 7 - chess.square_rank(square)
    )
    multiplier = rank_multiplier[rank]
    score *= multiplier

    return score


def center_control_score(board, user_side):
    score_user = 0
    score_opponent = 0

    for square in big_center:
        score_user += square_control_score(board, square, user_side)
        score_opponent += square_control_score(board, square, not user_side)

    return score_user, score_opponent, score_user - score_opponent


def get_open_files(board):
    open_files = []

    for file in range(8):
        has_pawn = False
        for rank in range(8):
            if board.piece_type_at(chess.square(file, rank)) == chess.PAWN:
                has_pawn = True
                break

        if not has_pawn:
            open_files.append(file)

    return open_files


def get_semi_open_files(board, color):
    semi_open_files = []

    for file in range(8):
        own_pawns = False
        opposite_pawns = False
        for rank in range(8):
            square = chess.square(file, rank)
            if board.piece_type_at(square) == chess.PAWN:
                if board.color_at(square) == color:
                    own_pawns = True
                else:
                    opposite_pawns = True

        if not own_pawns and opposite_pawns:
            semi_open_files.append(file)

    return semi_open_files


def file_control_score(board, user_side):
    open_files = get_open_files(board)
    user_semi_open_files = get_semi_open_files(board, user_side)
    opponent_semi_open_files = get_semi_open_files(board, not user_side)

    user_file_control_score = 0
    opponent_file_control_score = 0

    for file in open_files:
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_type_at(square)
            color = board.color_at(square)
            if piece in [chess.ROOK, chess.QUEEN]:
                defenders = len(board.attackers(color, square))
                if color == user_side:
                    user_file_control_score += file_control_piece_multiplier[piece] * (
                        1 + defenders)
                else:
                    opponent_file_control_score += file_control_piece_multiplier[piece] * (
                        1 + defenders)

    for file in user_semi_open_files:
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_type_at(square)
            color = board.color_at(square)
            if piece in [chess.ROOK, chess.QUEEN] and color == user_side:
                user_file_control_score += file_control_piece_multiplier[piece] * 0.5

    for file in opponent_semi_open_files:
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_type_at(square)
            color = board.color_at(square)
            if piece in [chess.ROOK, chess.QUEEN] and color != user_side:
                opponent_file_control_score += file_control_piece_multiplier[piece] * 0.5

    return (
        user_file_control_score,
        opponent_file_control_score,
        user_file_control_score - opponent_file_control_score,
    )


def pawn_structure(board):
    '''
    return 'open', 'semiopen', 'semiclosed', 'closed'

    only taking into account C, D, E, F files
    ram structure = two enemy pawns on the same file, blocking each other's forward movement

    number of pawns:
    7 - 8+- closed or semiclosed
    5 - 6 - semiclosed or semiopen
    3 - 4 - semiopen or open
    0 - 2 - open

    number of rams:
    4 - closed
    3 - closed or semiclosed
    2 - closed semiclosed or semiopen
    1 - semiopen or open
    '''
    num_pawns = 0
    for file in range(2, 6):
        for rank in range(1, 7):
            if board.piece_type_at(chess.square(file, rank)) == chess.PAWN:
                num_pawns += 1

    num_rams = 0
    for file in range(2, 6):
        for rank in range(1, 7):
            square = chess.square(file, rank)
            if (board.piece_type_at(square) == chess.PAWN and
                board.color_at(square) == chess.WHITE and
                board.piece_type_at(square + 8) == chess.PAWN and
                    board.color_at(square + 8) == chess.BLACK):
                num_rams += 1

    if num_pawns <= 2:
        return 'open'

    if num_pawns <= 4:
        if num_rams == 2:
            return 'semi-open'
        return 'open'

    if num_pawns == 5:
        return 'semi-open'

    if num_pawns == 6:
        if num_rams >= 2:
            return 'semi-closed'
        return 'semi-open'

    # if it gets here, num_pawns >= 7
    if num_rams >= 2:
        return 'closed'
    return 'semi-closed'
