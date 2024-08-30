from multiprocessing import cpu_count

from chess import Board
from stockfish import Stockfish

from utils.print_extra import print_with_timestamp

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
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.setup_stockfish()
        # print_with_timestamp(
        #     "Stockfish Parameters: " + str(self.stockfish.get_parameters())
        # )

    def setup_stockfish(self):
        self.stockfish = Stockfish(
            self.stockfish_path,
            depth=self.depth,
            parameters={"Hash": 1024, "Threads": cpu_count()},
        )

    def __getstate__(self):
        return self.stockfish_path, self.depth

    def __setstate__(self, state):
        (path, depth) = state
        self.stockfish_path = path
        self.depth = depth
        self.setup_stockfish()

    def evaluate_with_stockfish(
        self, position: Board, first: bool = False
    ) -> StockfishEvaluation:
        self.stockfish.set_fen_position(position.fen(), send_ucinewgame_token=first)
        return StockfishEvaluation(self.stockfish.get_evaluation())

    def get_best_move(self, position: Board, first: bool = False):
        self.stockfish.set_fen_position(position.fen(), send_ucinewgame_token=first)
        return self.stockfish.get_best_move()

    def prepare_stockfish(self, fen: str):
        self.stockfish.set_fen_position(fen, False)
        self.stockfish._go()

    def get_hacky_eval(self, fen):
        evaluation = dict()
        compare = 1 if "w" in fen else -1
        while True:
            text = self.stockfish._read_line()
            splitted_text = text.split(" ")
            if splitted_text[0] == "info":
                for n in range(len(splitted_text)):
                    if splitted_text[n] == "score":
                        evaluation = {
                            "type": splitted_text[n + 1],
                            "value": int(splitted_text[n + 2]) * compare,
                        }
            elif splitted_text[0] == "bestmove":
                return evaluation

    def get_prepared_evaluation(self, fen):
        return StockfishEvaluation(self.get_hacky_eval(fen))

    def reset_stockfish(self):
        self.stockfish._prepare_for_new_position()
