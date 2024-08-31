# Chess Imitation AI
This is my attempt of creating an AI model that learns how to play chess in *my style*. That includes:
- relative strength
- opening choice
- balance between positional and dynamic play

### Backlog of development progress

>### 25/04
>- [exported](https://github.com/mateipopescu1510/Chess-Imitation-AI/blob/main/src/export_games.py) lichess games in bulk as *.json* files
>- excluded games before 2021 and with less than 15 moves

>### 26/04
>- formatted games into *.csv* file
>- experimenting with [python-chess](https://python-chess.readthedocs.io/en/latest/), integrated own games with python-chess display
>- experimenting with [stockfish](https://pypi.org/project/stockfish/)
>- converted moves from *SAN* notation (```e4 c5 Nf3 d6```) to *UCI* notation (```e2e4 c7c5 g1f3 d7d6```) necessary for integration with chess engines

>### 29/04
>- generated [dataset](https://github.com/mateipopescu1510/Chess-Imitation-AI/blob/main/data_old/dataset_old.csv) with all individual positions from each game using the moves
>- paired each position with the move(s) I made from that point

>### 3/07
>- restructured files, included [all games](https://github.com/mateipopescu1510/Chess-Imitation-AI/tree/main/games_json) in JSON format as raw data
>- expanded [dataset](https://github.com/mateipopescu1510/Chess-Imitation-AI/blob/main/data_old/dataset_evalbefore.csv) by evaluating every position reached with stockfish

>### 5/07
>- expanded [dataset](https://github.com/mateipopescu1510/Chess-Imitation-AI/blob/main/data_old/dataset_evalafter.csv) by evaluating every position *before and after* my moves with stockfish
>- included game phase for every position (```opening```, ```middlegame``` or ```endgame```) determined with simple [heuristics](https://github.com/mateipopescu1510/Chess-Imitation-AI/blob/main/src/positional_features.py) (with more in progress)
>- converted move counter from each position into probability of a move being made
>- included further information such as [opening code](https://en.wikipedia.org/wiki/List_of_chess_openings) and average ELO diff in every position reached 

>### 8/07
>- restructured [dataset](https://github.com/mateipopescu1510/Chess-Imitation-AI/blob/main/data_old/dataset_kingdanger.csv) by including more columns for redundancy
>- added *king danger* data, using information and heuristics from the [Chess Programming Wiki](https://www.chessprogramming.org/King_Safety) as inspiration

>### 9/07
>- expanded [dataset](https://github.com/mateipopescu1510/Chess-Imitation-AI/blob/main/data/dataset.csv) by including data about *open and semi-open file control*, *center control* with my own simplistic scoring system
>- also added data about *pawn structure* - what I think to be one of the most important that determines how well I play a position; my style being more positional, with a preference for *closed* positions
>   - by counting pawns and [ram formations](https://www.chessprogramming.org/Pawn_Rams_(Bitboards)), every position was classified as ```open```, ```semi-open```, ```semi-closed``` or ```closed```.

>### 3/08
>- encoded data necessary to be used with neural network training (boards, moves)
>- created basic ResNet (residual neural network) with a variable number of residual layers

>### 14/08
>- reorganized files
>- implemented custom function to evaluate all positions in the dataset, to be used in the model's training
>- better move encoding and policy generation. enforced legal moves by using legal move masks both in training policies and model policy output
>- improved model architecture to return both a *policy* and an *evaluation*
>- created a search tree with *alpha beta pruning* to search based on the model's evaluation
>- implemented a simple interface in pygame