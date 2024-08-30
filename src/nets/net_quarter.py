from collections import namedtuple
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from dataloader.pgn_to_tensor import LabeledMove
from utils.print_extra import print_with_timestamp
from utils.stopwatch import Stopwatch

CollatedData = namedtuple("CollatedData", ["data", "labels"])

QuarteredData = namedtuple(
    "QuarteredData",
    [
        "one_quarter",
        "two_quarter",
        "three_quarter",
        "four_quarter",
    ],
)


class CuttlefishNetwork(nn.Module):

    # INPUT_LAYER_OUTPUT_SIZE = 512
    # # This used to be 512 as well
    # HIDDEN_LAYER_SIZE = 256
    # OUTPUT_LAYER_INPUT_SIZE = HIDDEN_LAYER_SIZE // 2

    INPUT_LAYER_OUTPUT_SIZE = 2048
    HIDDEN_LAYER_SIZE = 512
    OUTPUT_LAYER_INPUT_SIZE = 128

    def __init__(self):
        super(CuttlefishNetwork, self).__init__()

        self.q1_input_layer = nn.LazyLinear(self.INPUT_LAYER_OUTPUT_SIZE)
        self.q1_hidden_layer_1 = nn.Linear(
            self.INPUT_LAYER_OUTPUT_SIZE, self.HIDDEN_LAYER_SIZE
        )
        self.q1_hidden_layer_2 = nn.Linear(
            self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_INPUT_SIZE
        )
        self.q1_output_layer = nn.Linear(self.OUTPUT_LAYER_INPUT_SIZE, 1)

        self.q2_input_layer = nn.LazyLinear(self.INPUT_LAYER_OUTPUT_SIZE)
        self.q2_hidden_layer_1 = nn.Linear(
            self.INPUT_LAYER_OUTPUT_SIZE, self.HIDDEN_LAYER_SIZE
        )
        self.q2_hidden_layer_2 = nn.Linear(
            self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_INPUT_SIZE
        )
        self.q2_output_layer = nn.Linear(self.OUTPUT_LAYER_INPUT_SIZE, 1)

        self.q3_input_layer = nn.LazyLinear(self.INPUT_LAYER_OUTPUT_SIZE)
        self.q3_hidden_layer_1 = nn.Linear(
            self.INPUT_LAYER_OUTPUT_SIZE, self.HIDDEN_LAYER_SIZE
        )
        self.q3_hidden_layer_2 = nn.Linear(
            self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_INPUT_SIZE
        )
        self.q3_output_layer = nn.Linear(self.OUTPUT_LAYER_INPUT_SIZE, 1)

        self.q4_input_layer = nn.LazyLinear(self.INPUT_LAYER_OUTPUT_SIZE)
        self.q4_hidden_layer_1 = nn.Linear(
            self.INPUT_LAYER_OUTPUT_SIZE, self.HIDDEN_LAYER_SIZE
        )
        self.q4_hidden_layer_2 = nn.Linear(
            self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_INPUT_SIZE
        )
        self.q4_output_layer = nn.Linear(self.OUTPUT_LAYER_INPUT_SIZE, 1)

        # pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print_with_timestamp(f"N parameters = {pytorch_total_params}, estimated amount of data needed = {pytorch_total_params * 10}")

    @staticmethod
    def forward_for_quarter(
        move_tensor_list, input_layer, hidden_layer_1, hidden_layer_2, output_layer
    ):
        if len(move_tensor_list) == 0:
            return torch.tensor([], dtype=torch.float32)

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

        x = torch.cat(
            (
                friendly_pieces_before,
                unfriendly_pieces_before,
                friendly_pieces_after,
                unfriendly_pieces_after,
                evaluation_before,
                evaluation_after,
            ),
            dim=1,
        )

        x = input_layer(x)
        x = F.leaky_relu(x)

        x = hidden_layer_1(x)
        x = F.leaky_relu(x)

        x = hidden_layer_2(x)
        x = F.leaky_relu(x)

        return output_layer(x)

    def forward(self, data: QuarteredData):
        q1 = self.forward_for_quarter(
            data.one_quarter,
            self.q1_input_layer,
            self.q1_hidden_layer_1,
            self.q1_hidden_layer_2,
            self.q1_output_layer,
        )

        q2 = self.forward_for_quarter(
            data.two_quarter,
            self.q2_input_layer,
            self.q2_hidden_layer_1,
            self.q2_hidden_layer_2,
            self.q2_output_layer,
        )

        q3 = self.forward_for_quarter(
            data.three_quarter,
            self.q3_input_layer,
            self.q3_hidden_layer_1,
            self.q3_hidden_layer_2,
            self.q3_output_layer,
        )

        q4 = self.forward_for_quarter(
            data.four_quarter,
            self.q4_input_layer,
            self.q4_hidden_layer_1,
            self.q4_hidden_layer_2,
            self.q4_output_layer,
        )

        res = torch.cat([q1, q2, q3, q4])
        return res


class Collator:
    def __init__(self):
        self.timer = Stopwatch()
        self.timer.start()
        self.q1_n = 0
        self.q2_n = 0
        self.q3_n = 0
        self.q4_n = 0

    def custom_collate(self, move_tensors: List[LabeledMove]) -> CollatedData:
        one_quarter = []
        two_quarter = []
        three_quarter = []
        four_quarter = []

        for move_tensor in move_tensors:
            n = move_tensor.move.nPiecesBefore / 32
            if n <= 0.25:
                one_quarter.append(move_tensor)
                self.q1_n += 1
            elif n <= 0.5:
                two_quarter.append(move_tensor)
                self.q2_n += 1
            elif n <= 0.75:
                three_quarter.append(move_tensor)
                self.q3_n += 1
            else:
                four_quarter.append(move_tensor)
                self.q4_n += 1

        if self.timer.has_minute_elapsed():
            print_with_timestamp(
                f"Have seen the following move counts: [ {self.q1_n:,} | {self.q2_n:,} | {self.q3_n:,} | {self.q4_n:,} ]"
            )
            self.timer.reset()

        def get_move(t):
            return t.move

        labels = []
        for a in one_quarter:
            labels.append(a.is_human)
        for a in two_quarter:
            labels.append(a.is_human)
        for a in three_quarter:
            labels.append(a.is_human)
        for a in four_quarter:
            labels.append(a.is_human)

        return CollatedData(
            QuarteredData(
                list(map(get_move, one_quarter)),
                list(map(get_move, two_quarter)),
                list(map(get_move, three_quarter)),
                list(map(get_move, four_quarter)),
            ),
            torch.tensor([[label] for label in labels], dtype=torch.float32),
        )
