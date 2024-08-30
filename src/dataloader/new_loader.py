import io
from typing import Generator

import torch
import zstandard
from chess.pgn import Game, read_game
from torch.utils.data import IterableDataset

from dataloader.pgn_to_tensor import PgnMoveTensorExtractor


class DirectFromPgnLoader(IterableDataset):
    def __init__(self, pgns, game_length_min=0):
        self.pgns = pgns
        self.pgn_move_tensor_extractor = PgnMoveTensorExtractor(
            stockfish_path="/Users/grego/stockfish-windows-x86-64-avx512/stockfish/stockfish-windows-x86-64-avx512.exe",
            game_length_min=game_length_min,
        )

    @staticmethod
    def pgn_iterator(data) -> Generator[Game, None, None]:
        while True:
            pgn = read_game(data)
            if pgn is None:
                return
            yield pgn

    def generator_iterator(self) -> Generator[Game, None, None]:
        pgns = self.pgns
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if worker_info.num_workers > len(self.pgns):
                raise ValueError(
                    "The number of workers must match the number of pgn files"
                )
            pgns = [self.pgns[worker_info.id]]

        for pgn_file in pgns:
            with open(pgn_file, "rb") as fh:
                dctx = zstandard.ZstdDecompressor(max_window_size=2**30)
                stream_reader = dctx.stream_reader(fh)
                text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
                for pgn in self.pgn_iterator(text_stream):
                    try:
                        for (
                            tensor
                        ) in self.pgn_move_tensor_extractor.extract_move_tensors(pgn):
                            yield tensor
                    except Exception as e:
                        print(
                            f"Had an exception processing PGN {self.pgn_move_tensor_extractor.pgns_serialized}: {e}"
                        )

    def __iter__(self):
        return iter(self.generator_iterator())


class CircularDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.iter = iter(dataset)

    def __getstate__(self):
        return self.dataset

    def __setstate__(self, state):
        self.dataset = state
        self.iter = iter(self.dataset)

    def __iter__(self):
        def generator():
            while True:
                i = next(self.iter, None)
                if i is None:
                    self.iter = iter(self.dataset)
                    yield next(self.iter)
                else:
                    yield i

        return iter(generator())


if __name__ == "__main__":
    for x in DirectFromPgnLoader(["data/lichess_db_standard_rated_2024-04.pgn.zst"]):
        print(x)
        break
