# Chess Imitation AI
This is my attempt of creating an AI model that learns how to play chess in *my style*. That includes:
- relative strength
- opening choice
- balance between positional and dynamic play

Development is still in its early stages, with progress documented at each step.
### Backlog of development progress

>### 25/04
>
>- exported lichess games in bulk as *.json* files
>- excluded games before 2021 and with less than 15 moves

>### 26/04
>
>- formatted games into *.csv* file
>- experimenting with [python-chess](https://python-chess.readthedocs.io/en/latest/), integrated own games with python-chess display
>- experimenting with [stockfish](https://pypi.org/project/stockfish/)
>- converted moves from *SAN* notation (```e4 c5 Nf3 d6```) to *UCI* notation (```e2e4 c7c5 g1f3 d7d6```) necessary for integration with chess engines

>### 29/04
>
>- generated all individual positions from each game using the moves
>- paired each position with the move(s) I made from that point
