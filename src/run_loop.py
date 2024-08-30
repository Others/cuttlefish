import math
import random
from collections import namedtuple
from random import choices
from typing import List

import chess
import torch
from chess import Board, InvalidMoveError, Move
from torch import sigmoid
from torchviz import make_dot

from dataloader.pgn_to_tensor import LabeledMove, MoveTensorCreator
from nets.net_4096 import Collator, CuttlefishNetwork
from stockfish_interface.stockfish_wrapper import StockfishEvaluator

EvaluatedMove = namedtuple("EvaluatedMove", ["move", "evaluation", "humanness"])


class Cuttlefish:
    def __init__(self, stockfish_path, weights):
        # FIXME: We currently use depth 5 for ease of training data generation
        # When we expand cuttlefish to elo bands, we'll need better data to play at a high level
        self.stockfish_evaluator = StockfishEvaluator(stockfish_path, depth=5)
        self.model = CuttlefishNetwork()
        self.model.load_state_dict(torch.load(weights))
        self.move_tensor_creator = MoveTensorCreator()
        self.collator = Collator()
        self.model.eval()

    def pick_move(self, current_position: Board):
        # We set "first" to true here to reset stockfish_interface cache
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

        humanness = list(map(lambda x: x.humanness, evaluations))

        max_humanness = max(humanness)
        variance = max_humanness - min(humanness)
        if variance < 0.01:
            # if variance > 0:
            #     print("The variance between moves is negligible, playing the best option rather than using probablity.")
            #     return max(evaluations, key=lambda x: x.humanness).move
            # else:
            print(
                "The variance between moves is near zero. Playing approximate best move instead"
            )
            return current_position.parse_san(best_move)

        # normalized_humanness = normalization(humanness)
        # for idx, evaluation in enumerate(evaluations):
        #     print(
        #         f"Move {current_position.san(evaluation.move)} has score {normalized_humanness[idx] * 100:.2f}% (normalized from {evaluation.humanness})"
        #     )
        # print(f"Best Move: {best_move}")
        #
        # # Pick one
        # n_evaluations = len(evaluations)
        # choice = choices(evaluations, normalized_humanness)[0]
        # # choice = max(evaluations, key=lambda x: x.humanness)
        choice = tournament_selection(current_position, evaluations)
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

        # We need to re-collate here to ensure parity with the original training process
        collated = self.collator.custom_collate([LabeledMove(tensors, False)])

        prob = sigmoid(self.model(collated.data))
        # print(prob)
        return prob[0].item()

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


def tournament_selection(current_position, evaluated_moves: List[EvaluatedMove]):
    def bradley_miller_score(move):
        # We assume that a random move has a strength score of 1
        #
        # Therefore we have:
        # P(random move > this move) = 1 / (score_move + 1)
        # And from our binary classifier we have:
        # humanness = 1 - P(random move > this move)
        # Rearranging we have:
        # (1 - humanness) (score_move + 1) = 1
        # score_move = (1 / (1 - humanness)) - 1

        # But we need to avoid division by zero, so we do this
        if (1 - move.humanness) == 0.0:
            return math.inf

        return (1 / (1 - move.humanness)) - 1

    def bradley_miller_probability(a, b):
        if (a + b) == 0:
            if a > b:
                return 1.0
            else:
                return 0.0

        return a / (a + b)

    def bradley_miller_defeat(a, b, rounds):
        a_score = bradley_miller_score(a)
        b_score = bradley_miller_score(b)

        a_defeats = bradley_miller_probability(a_score, b_score)

        print(f"Defeat probability {a_defeats:.4f}, doing best of {rounds}")

        a_wins = 0
        for _ in range(rounds):
            if a_defeats > random.random():
                a_wins += 1
        return a_wins >= rounds / 2

    # Now we run a gauntlet, where the weakest moves must beat all the other moves
    moves_lowest_to_highest = list(
        sorted(evaluated_moves, key=lambda m: m.humanness, reverse=True)
    )
    best_move = moves_lowest_to_highest.pop()
    while len(moves_lowest_to_highest) > 0:
        new_move = moves_lowest_to_highest.pop()
        print(
            f"Current best move {best_move.move} (score: {best_move.humanness:.4f}). Considering {new_move.move} (score {new_move.humanness:.4f})"
        )
        if bradley_miller_defeat(new_move, best_move, 1):
            best_move = new_move

    return best_move

    # sorted_moves = list(sorted(evaluated_moves, key=lambda x: x.humanness, reverse=True))
    # for move in sorted_moves:
    #     print(f"Move {current_position.san(move.move)} has a {move.humanness * 100:.2f}% chance of being human")
    # for move in sorted_moves:
    #     if move.humanness > random.uniform(0, 1):
    #         return move
    # return sorted_moves[0]


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
    cuttlefish = Cuttlefish(
        "/Users/grego/stockfish-windows-x86-64-avx512/stockfish/stockfish-windows-x86-64-avx512.exe",
        "best_model.pth",
    )
    # cuttlefish = Cuttlefish("/Users/grego/stockfish-windows-x86-64-avx512/stockfish/stockfish-windows-x86-64-avx512.exe", "best_model_819.pth")
    # board = Board("8/2p5/3k4/3r4/8/8/5PP1/6K1 w - - 0 1")
    board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    while not board.is_checkmate():
        print_board(board)
        print(board.fen(), "\n")

        legal_moves = [f"{board.san(move)}:{move.uci()}" for move in board.legal_moves]
        print("Legal Moves", legal_moves)
        if len(legal_moves) == 0:
            print("The game is over. Hopefully we won.")
            return

        while True:
            try:
                human_input = input()
                human_move = Move.from_uci(human_input)
                board.push(human_move)
                break
            except InvalidMoveError:
                print("Illegal Move!")
            except AssertionError:
                print("Illegal Move!")

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
