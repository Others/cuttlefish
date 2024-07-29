from multiprocessing import cpu_count

from chess import Board
from stockfish import Stockfish

from print_with_timestamp import print_with_timestamp

MAX_EVAL = 20000


class StockfishEvaluation:
    def __init__(self, dict_evaluation):
        given_value = dict_evaluation["value"]
        if dict_evaluation["type"] == "mate":
            # This represents mate as a high number of centipawns
            given_value *= MAX_EVAL
        elif dict_evaluation["type"] == "cp":
            pass
        else:
            raise ValueError(f"Evaluation is not valid: {dict_evaluation}")

        # Force the value into a centipawn range
        self.value = max(min(given_value, MAX_EVAL), -MAX_EVAL)


class StockfishEvaluator:
    def __init__(self, stockfish_path, depth=15):
        self.stockfish = Stockfish(
            stockfish_path,
            depth=depth,
            parameters={"Hash": 1024, "Threads": cpu_count()},
        )
        print_with_timestamp(
            "Stockfish Parameters: " + str(self.stockfish.get_parameters())
        )

    def evaluate_with_stockfish(
        self, position: Board, first: bool = False
    ) -> StockfishEvaluation:
        self.stockfish.set_fen_position(position.fen(), send_ucinewgame_token=first)
        return StockfishEvaluation(self.stockfish.get_evaluation())

    def get_best_move(self, position: Board, first: bool = False):
        self.stockfish.set_fen_position(position.fen(), send_ucinewgame_token=first)
        return self.stockfish.get_best_move()
