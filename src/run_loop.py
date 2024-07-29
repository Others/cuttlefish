from collections import namedtuple
from random import choices, uniform

import chess
import torch
from chess import Board, InvalidMoveError, Move
from torch import sigmoid
from torchviz import make_dot

from pgn_to_tensor import MoveTensorCreator
from stockfish_wrapper import StockfishEvaluator
from train_net import CuttlefishNetwork


class Cuttlefish:
    def __init__(self, stockfish_path, weights):
        # FIXME: We currently use depth 5 for ease of training data generation
        # When we expand cuttlefish to elo bands, we'll need better data to play at a high level
        self.stockfish_evaluator = StockfishEvaluator(stockfish_path, depth=5)
        self.model = CuttlefishNetwork()
        self.model.load_state_dict(torch.load(weights))
        self.move_tensor_creator = MoveTensorCreator()
        self.model.eval()

    def pick_move(self, current_position: Board):
        # We set "first" to true here to reset stockfish cache
        original_position = current_position.copy()
        original_position_evaluation = self.stockfish_evaluator.evaluate_with_stockfish(
            current_position, True
        )
        best_move = self.stockfish_evaluator.get_best_move(original_position)
        print(f"Cuttlefish Evaluation: {original_position_evaluation.value}")

        # Check all evaluations
        moves = list(current_position.legal_moves)
        if len(moves) == 0:
            return None

        evaluations = list()
        for move in moves:
            current_position.push(move)
            evaluation = self._evaluate(
                original_position, original_position_evaluation, move, current_position
            )
            evaluations.append(evaluation)
            current_position.pop()

        humanness = list(map(lambda x: x.humanness[0], evaluations))

        normalized_humanness = normalization(humanness)
        for idx, evaluation in enumerate(evaluations):
            print(
                f"Move {current_position.san(evaluation.move)} has score {normalized_humanness[idx] * 100:.2f}% (normalized from {evaluation.humanness})"
            )

        variance = max(humanness) - min(humanness)
        if variance < 0.01:
            if variance > 0:
                print("The variance between moves is negligible, playing the best option rather than using probablity.")
                return max(evaluations, key=lambda x: x.humanness).move
            else:
                print(
                    "The variance between moves is zero. Playing approximate best move instead"
                )
                return current_position.parse_san(best_move)

        # Pick one
        n_evaluations = len(evaluations)
        choice = choices(evaluations, normalized_humanness)[0]
        # choice = max(evaluations, key=lambda x: x.humanness)
        return choice.move

    def _evaluate(
        self,
        previous_position: Board,
        previous_position_stockfish_evaluation,
        move: Move,
        position_after_move: Board,
    ):
        stockfish_evaluation = self.stockfish_evaluator.evaluate_with_stockfish(
            position_after_move, False
        )
        humanness = self._evaluate_humanness(
            previous_position,
            previous_position_stockfish_evaluation,
            position_after_move,
            stockfish_evaluation,
        )
        return EvaluatedMove(move, stockfish_evaluation, humanness)

    def _evaluate_humanness(
        self,
        previous_position: Board,
        previous_position_stockfish_evaluation,
        new_position: Board,
        new_position_stockfish_evaluation,
    ):
        tensors = self.move_tensor_creator.move_to_tensors(
            previous_position,
            previous_position_stockfish_evaluation,
            new_position,
            new_position_stockfish_evaluation,
        )
        # print(tensors)
        prob = sigmoid(self.model([tensors]))
        # print(prob)
        return prob[0]

    def visualize(
        self,
        previous_position: Board,
        previous_position_stockfish_evaluation,
        new_position: Board,
        new_position_stockfish_evaluation,
    ):
        tensors = self.move_tensor_creator.move_to_tensors(
            previous_position,
            previous_position_stockfish_evaluation,
            new_position,
            new_position_stockfish_evaluation,
        )
        y = self.model([tensors])
        dot = make_dot(y.mean(), params=dict(self.model.named_parameters()))
        dot.render("visualized_graph", format="png")


def normalization(float_list):
    alpha = 8
    exp_list = [x**alpha for x in float_list]
    total = sum(exp_list)
    # Edge case -- only possible if all values are zero
    if total == 0:
        return [1.0 / len(float_list)] * len(float_list)
    return [x / total for x in exp_list]


EvaluatedMove = namedtuple("EvaluatedMove", ["move", "evaluation", "humanness"])

unicode_pieces = {
    "p": "♙",
    "n": "♘",
    "b": "♗",
    "r": "♖",
    "q": "♕",
    "k": "♔",
    "P": "♟",
    "N": "♞",
    "B": "♝",
    "R": "♜",
    "Q": "♛",
    "K": "♚",
    ".": "·",
}


# Print the board
def print_board(board):
    for rank in range(8, 0, -1):
        line = str(rank) + " "
        for file in range(1, 9):
            piece = board.piece_at(chess.square(file - 1, rank - 1))
            line += unicode_pieces[piece.symbol()] if piece else unicode_pieces["."]
            line += " "
        print(line)
    print("  a b c d e f g h")


def main():
    cuttlefish = Cuttlefish("/usr/local/bin/stockfish", "best_model.pth")
    board = Board("8/2p5/3k4/3r4/8/8/5PP1/6K1 w - - 0 1")
    # board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    while not board.is_checkmate():
        print_board(board)
        print(board.fen(), "\n")

        legal_moves = [f"{board.san(move)}:{move.uci()}" for move in board.legal_moves]
        print("Legal Moves", legal_moves)
        if len(legal_moves) == 0:
            print("The game is over. Hopefully we won.")
            return
        human_move = None
        while human_move is None:
            try:
                human_input = input()
                human_move = Move.from_uci(human_input)
            except InvalidMoveError:
                print("Illegal Move!")
            except AssertionError:
                print("Illegal Move!")
        board.push(human_move)

        cuttlefish_move = cuttlefish.pick_move(board)
        if cuttlefish_move is None:
            print("The game is over. Hopefully we won.")
            return
        print(
            "Cuttlefish plays:", board.san(cuttlefish_move), "(", cuttlefish_move, ")"
        )
        board.push(cuttlefish_move)


if __name__ == "__main__":
    main()
