import berserk
import chess
import json
import pandas as pd
import os
import re
import csv

USER = "matei_popescu1510"
JSON_DIR_PATH = "./games_json/"
CSV_FILE_PATH = "./data/game_data.csv"

client = berserk.Client()
games = list(
    client.games.export_by_player(
        USER,
        rated=True,
        perf_type="bullet,blitz,rapid",
        sort="dateAsc",
        pgn_in_json=True,
        # max=10,
    )
)

for game in games:
    if "15. " not in game["pgn"] or game["createdAt"].year < 2021:
        continue
    game_id = JSON_DIR_PATH + game["id"] + ".json"
    # print(game_id)
    with open(game_id, "w") as path:
        json.dump(game, path, default=str, sort_keys=True)


def extract_opening(pgn):
    opening = ""
    pattern = r"\[ECO \"([^\"]+)\"\]"
    match = re.search(pattern, pgn)
    if match:
        opening = match.group(1)
    return opening


games = []
fieldnames = [
    "game_id",
    "white_id",
    "white_elo",
    "black_id",
    "black_elo",
    "opening",
    "moves",
    "winner",
]

for filename in os.listdir(JSON_DIR_PATH):
    with open(os.path.join(JSON_DIR_PATH, filename), "r") as game_json:
        game = json.load(game_json)

        game_id = game["id"]
        white_id = game["players"]["white"]["user"]["id"]
        white_elo = game["players"]["white"]["rating"]
        black_id = game["players"]["black"]["user"]["id"]
        black_elo = game["players"]["black"]["rating"]
        opening = extract_opening(game["pgn"])
        moves = game["moves"]

        try:
            winner = game["winner"]
        except KeyError:
            winner = "draw"

        data = {
            "game_id": game_id,
            "white_id": white_id,
            "white_elo": white_elo,
            "black_id": black_id,
            "black_elo": black_elo,
            "opening": opening,
            "moves": moves,
            "winner": winner,
        }
        games.append(data)

with open(CSV_FILE_PATH, "w") as game_data:
    writer = csv.DictWriter(game_data, fieldnames=fieldnames)
    writer.writeheader()
    for game in games:
        writer.writerow(game)


data = pd.read_csv(CSV_FILE_PATH)

for index, row in data.iterrows():
    moves = row["moves"].split()
    uci_moves = []

    board = chess.Board()

    for move in moves:
        uci_moves.append(str(board.push_san(move)))

    data.loc[index, "moves"] = " ".join(uci_moves)


data.to_csv(CSV_FILE_PATH)
