import chess
import pandas as pd

CSV_FILE_PATH = "./game_data.csv"

data = pd.read_csv(CSV_FILE_PATH)

for index, row in data.iterrows():
    moves = row["moves"].split()
    uci_moves = []

    board = chess.Board()

    for move in moves:
        uci_moves.append(str(board.push_san(move)))

    data.loc[index, "moves"] = " ".join(uci_moves)


data.to_csv(CSV_FILE_PATH)
