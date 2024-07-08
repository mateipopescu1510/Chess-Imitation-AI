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
delta_king_zone_A8H1 = [-10, -9, -8, -2, -1, 0, 6, 7, 8, 14, 15, 16, 23, 24, 31, 32]
delta_king_zone_A1H8 = [-8, -7, -6, 0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 32, 33]

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
            score += attack_values[board.piece_type_at(attacker)] * proximity_multiplier

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
    user_king_danger = king_zone_attackers_score(board, user_side) * pawn_shield_score(
        board, user_side
    )
    opponent_king_danger = king_zone_attackers_score(
        board, not user_side
    ) * pawn_shield_score(board, not user_side)
    return (
        round(user_king_danger, 2),
        round(opponent_king_danger, 2),
        round(opponent_king_danger - user_king_danger, 2),
    )


def pawn_structure(board, color):
    return None


def piece_activity(board, color):
    return None


def center_control_score(board, color):
    return None


def open_file_control(board, color):
    return None


## etc
# board = chess.Board("r4rk1/ppq1bppp/2p1pn2/6B1/3P4/2PB1Q1P/PP3PP1/R4R1K b - - 6 16")
# king_zone = chess.SquareSet(get_king_zone(board, chess.WHITE))
# print(king_zone)
# for square in king_zone:
#     print(chess.square_name(square), end=" ")

# print()
# print(king_danger_score(board, chess.BLACK))
# print(chess.H1 - chess.G4)


# print(pawn_shield_multiplier[pawn_shield_score(board, chess.BLACK)])
