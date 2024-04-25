import berserk
import datetime
import json

client = berserk.Client()


# start = berserk.utils.to_millis(datetime(2021, 1, 1))

# end = berserk.utils.to_millis(datetime(2024, 4, 24))

games = list(
    client.games.export_by_player(
        "Matei_Popescu1510",
        max=10,
        rated=True,
        perf_type="bullet,blitz,rapid",
        sort="dateAsc",
        pgn_in_json=True,
    )
)
print(games[1])

path = "./games/test.json", "w"
with open("./games/test.json", "w") as path:
    json.dump(games[1], path, default=str, sort_keys=True)


# TODO exclude games with less than 15 moves
# TODO exclude games earlier than 2021
print("15" in games[1]["pgn"])
print(games[1]["createdAt"].year > 2020)
# for index, game in enumerate(games):
#     path = "./games/game" + str(index) + ".pgn"
#     g = open(path, "w", encoding="utf8")
#     g.write(game)
