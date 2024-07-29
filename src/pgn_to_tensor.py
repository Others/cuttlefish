import io
import random
import subprocess
import sys
from collections import namedtuple
from typing import Generator, List

import numpy as np
import torch
from chess import BLACK, PIECE_TYPES, SQUARES, Board, Color
from chess.pgn import Game, read_game
from torch import Tensor, from_numpy, tensor

from stockfish_wrapper import StockfishEvaluation, StockfishEvaluator
from stopwatch import Stopwatch

PIECES = len(PIECE_TYPES)

# To have a board tensor for one color, we have one byte for every combination of
# - square number
# - piece type
# And we ignore pieces of the other color
ONE_COLOR_BOARD_TENSOR_LEN = len(SQUARES) * PIECES

MoveTensor = namedtuple(
    "MoveTensor",
    [
        # bytes of ONE_COLOR_BOARD_TENSOR_LEN
        "friendlyPiecesBefore",
        # bytes of ONE_COLOR_BOARD_TENSOR_LEN
        "unfriendlyPiecesBefore",
        # vector of a single signed float, positive if friendly is winning
        "evaluationBefore",
        # bytes of ONE_COLOR_BOARD_TENSOR_LEN
        "friendlyPiecesAfter",
        # bytes of ONE_COLOR_BOARD_TENSOR_LEN
        "unfriendlyPiecesAfter",
        # vector of a single signed float, positive if friendly is winning
        "evaluationAfter",
    ],
)

LabeledMove = namedtuple(
    "LabeledMove",
    [
        # type is `MoveTensor`
        "move",
        # a vector of a single float, either zero or non-zero
        # zero = non-human
        "is_human",
    ],
)


class MoveTensorCreator:
    def __init__(self):
        # This is for debugging / progress tracking
        self.moves_serialized = 0

    def move_to_tensors(
        self,
        old_board: Board,
        old_evaluation: StockfishEvaluation,
        new_board: Board,
        new_evaluation: StockfishEvaluation,
    ):
        self.moves_serialized += 1
        return MoveTensor(
            friendlyPiecesBefore=self.board_to_tensor_for_color(
                old_board, old_board.turn
            ),
            unfriendlyPiecesBefore=self.board_to_tensor_for_color(
                old_board,
                # `chess` uses a boolean for white vs black, so we can just invert that
                not old_board.turn,
            ),
            evaluationBefore=self.evaluation_to_tensor(old_board, old_evaluation),
            friendlyPiecesAfter=self.board_to_tensor_for_color(
                new_board, new_board.turn
            ),
            unfriendlyPiecesAfter=self.board_to_tensor_for_color(
                new_board,
                # `chess` uses a boolean for white vs black, so we can just invert that
                not new_board.turn,
            ),
            evaluationAfter=self.evaluation_to_tensor(new_board, new_evaluation),
        )

    @staticmethod
    def board_to_tensor_for_color(board: Board, target_color: Color) -> Tensor:
        board_tensor = np.zeros(ONE_COLOR_BOARD_TENSOR_LEN, dtype=np.uint8)
        for i in range(64):
            piece = board.piece_at(i)
            if piece is None or piece.color != target_color:
                continue
            # We subtract 1 because the "chess" module starts piece number assignments at 1
            piece_offset = piece.piece_type - 1
            board_tensor[i * PIECES + piece_offset] = 1
        return from_numpy(board_tensor)

    @staticmethod
    def evaluation_to_tensor(board: Board, evaluation: StockfishEvaluation) -> Tensor:
        if board.turn == BLACK:
            return tensor([evaluation.value * -1], dtype=torch.float32)
        return tensor([evaluation.value], dtype=torch.float32)


class PgnMoveTensorExtractor:
    def __init__(self, stockfish_path):
        self.move_tensor_creator = MoveTensorCreator()
        # FIXME: We currently use depth 5 for ease of training data generation
        # When we expand cuttlefish to elo bands, we'll need better data to play at a high level
        self.stockfish_wrapper = StockfishEvaluator(stockfish_path, depth=5)
        # This is for debugging / progress tracking
        self.pgns_serialized = 0

    # This extracts a move tensor for the real moves, and one fake move per position
    def extract_move_tensors(self, pgn: Game) -> List[LabeledMove]:
        tensors = []
        board = pgn.board()
        board_evaluation = self.stockfish_wrapper.evaluate_with_stockfish(
            board, first=True
        )

        self.pgns_serialized += 1

        for move in pgn.mainline_moves():
            old_board = board.copy()
            old_board_evaluation = board_evaluation

            board.push(move)
            true_move_board = board
            true_move_evaluation = self.stockfish_wrapper.evaluate_with_stockfish(
                true_move_board
            )
            true_move_tensor = self.move_tensor_creator.move_to_tensors(
                old_board, old_board_evaluation, true_move_board, true_move_evaluation
            )
            tensors.append(LabeledMove(true_move_tensor, human_vector(True)))

            # FIXME: If only one legal move, then skip all tensor appending

            other_moves = list(filter(lambda x: x != move, old_board.legal_moves))
            if len(other_moves) > 0:
                other_move = random.choice(other_moves)

                alt_board = old_board.copy()
                alt_board.push(other_move)
                alt_board_evaluation = self.stockfish_wrapper.evaluate_with_stockfish(
                    alt_board
                )
                alt_board_tensor = self.move_tensor_creator.move_to_tensors(
                    old_board, old_board_evaluation, alt_board, alt_board_evaluation
                )
                tensors.append(LabeledMove(alt_board_tensor, human_vector(False)))
        return tensors


def file_reader(pgn_bz2_file):
    try:
        # Run the bzcat command and capture the output
        process = subprocess.Popen(
            ["bzcat", pgn_bz2_file],
            stdout=subprocess.PIPE,
            stderr=sys.stderr.fileno(),
        )

        # Read the output line by line
        return io.TextIOWrapper(process.stdout, encoding="utf-8")
    except FileNotFoundError as e:
        print(f"bzcat command not found: {e}")
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def pgn_iterator(data) -> Generator[Game, None, None]:
    while True:
        pgn = read_game(data)
        if pgn is None:
            return
        yield pgn


def write_tensors(folder, tensors: List[LabeledMove], n):
    data = []
    label = []
    for d, l in tensors:
        data.append(d)
        label.append(l)
    torch.save({"data": data, "labels": label}, f"{folder}/{n}_pgns_tensorified.pt")


def human_vector(is_human: bool):
    return tensor([int(is_human)], dtype=torch.float32)


def main():
    print("Starting Up")
    pgn_move_tensor_extractor = PgnMoveTensorExtractor(
        stockfish_path="/usr/local/bin/stockfish"
    )
    stopwatch = Stopwatch()
    stopwatch.start()

    print("Starting PGN Processing...\n")
    pgn_list = file_reader("data/lichess_db_standard_rated_2020-10.pgn.bz2")
    tensors = []
    for pgn in pgn_iterator(pgn_list):
        try:
            tensors += pgn_move_tensor_extractor.extract_move_tensors(pgn)
        except Exception as e:
            print(
                f"Had an exception processing PGN {pgn_move_tensor_extractor.pgns_serialized}: {e}"
            )
        if stopwatch.has_minute_elapsed():
            pgn_count = pgn_move_tensor_extractor.pgns_serialized
            move_count = pgn_move_tensor_extractor.move_tensor_creator.moves_serialized

            stopwatch.reset()
            write_tensors("data3", tensors, move_count)
            tensors = []
            print(
                f"Update After a Minute:\n\tGames: {pgn_count}\n\tMoves: {move_count}\n"
            )


if __name__ == "__main__":
    main()
