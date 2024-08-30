from collections import namedtuple
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from dataloader.pgn_to_tensor import LabeledMove
from utils.stopwatch import Stopwatch

CollatedData = namedtuple("CollatedData", ["data", "labels"])


class CuttlefishNetwork(nn.Module):

    INPUT_LAYER_OUTPUT_SIZE = 4096
    HIDDEN_LAYER_SIZE = 1024
    OUTPUT_LAYER_INPUT_SIZE = 256

    def __init__(self):
        super(CuttlefishNetwork, self).__init__()

        self.input_layer = nn.LazyLinear(self.INPUT_LAYER_OUTPUT_SIZE)
        self.hidden_layer_1 = nn.Linear(
            self.INPUT_LAYER_OUTPUT_SIZE, self.HIDDEN_LAYER_SIZE
        )
        self.hidden_layer_2 = nn.Linear(
            self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_INPUT_SIZE
        )
        self.output_layer = nn.Linear(self.OUTPUT_LAYER_INPUT_SIZE, 1)

    def forward(self, tensor):
        x = self.input_layer(tensor)
        x = F.leaky_relu(x)

        x = self.hidden_layer_1(x)
        x = F.leaky_relu(x)

        x = self.hidden_layer_2(x)
        x = F.leaky_relu(x)

        x = self.output_layer(x)
        return x


class Collator:
    def __init__(self):
        self.timer = Stopwatch()
        self.timer.start()

    def custom_collate(self, move_tensors: List[LabeledMove]) -> CollatedData:
        move_tensor_list = list(map(lambda x: x.move, move_tensors))
        lables = torch.tensor(
            list(map(lambda x: [x.is_human], move_tensors)), dtype=torch.float32
        )

        last_move_square = torch.stack(
            [move.lastMoveSquare for move in move_tensor_list]
        )
        friendly_pieces_before = torch.stack(
            [move.friendlyPiecesBefore.to(torch.float) for move in move_tensor_list]
        )
        unfriendly_pieces_before = torch.stack(
            [move.unfriendlyPiecesBefore.to(torch.float) for move in move_tensor_list]
        )
        friendly_pieces_after = torch.stack(
            [move.friendlyPiecesAfter.to(torch.float) for move in move_tensor_list]
        )
        unfriendly_pieces_after = torch.stack(
            [move.unfriendlyPiecesAfter.to(torch.float) for move in move_tensor_list]
        )
        evaluation_before = torch.stack(
            [move.evaluationBefore for move in move_tensor_list]
        )
        evaluation_after = torch.stack(
            [move.evaluationAfter for move in move_tensor_list]
        )

        return CollatedData(
            torch.cat(
                (
                    last_move_square,
                    friendly_pieces_before,
                    unfriendly_pieces_before,
                    friendly_pieces_after,
                    unfriendly_pieces_after,
                    evaluation_before,
                    evaluation_after,
                ),
                dim=1,
            ),
            lables,
        )
