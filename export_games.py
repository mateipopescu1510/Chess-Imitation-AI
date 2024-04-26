import berserk

import json
import pandas as pd
import os

USER = "matei_popescu1510"
JSON_DIR_PATH = "./games_json/"
CSV_DIR_PATH = "./games_csv/"

client = berserk.Client()

games = list(
    client.games.export_by_player(
        USER,
        max=10,
        rated=True,
        perf_type="bullet,blitz,rapid",
        # sort="dateAsc",
        pgn_in_json=True,
    )
)

for game in games:
    if "15. " not in game["pgn"] or game["createdAt"].year < 2021:
        continue
    game_id = JSON_DIR_PATH + game["id"] + ".json"
    # print(game_id)
    with open(game_id, "w") as path:
        json.dump(game, path, default=str, sort_keys=True)


# for game in games:
#     if "15" not in game["pgn"] or game["createdAt"].year < 2021:
#         continue
#     id = game["id"]
#     moves = game["moves"]
#     pgn = game["pgn"]
#     white_id = game["players"]["white"]["user"]["id"]
#     white_elo = game["players"]["white"]["rating"]
#     black_id = game["players"]["black"]["user"]["id"]
#     black_elo = game["players"]["black"]["rating"]
#     winner = game["winner"]
#     print(id, white_id, white_elo, black_id, black_elo, winner)


# TODO turn games into CSV containing id, moves, pgn, blackId, blackRating, whiteId, whiteRating, winner
# TODO extract opening code from PGN to put in CSV

for filename in os.listdir(JSON_DIR_PATH):
    with open(os.path.join(JSON_DIR_PATH, filename), "r") as game_json:
        game = json.load(game_json)
        print(game["pgn"])

    break
