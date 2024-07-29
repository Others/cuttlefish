from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prettytable import PrettyTable
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data_loader import create_lazy_loaders, custom_collate
from pgn_to_tensor import MoveTensor
from print_with_timestamp import print_with_timestamp
from stopwatch import Stopwatch


class CuttlefishNetwork(nn.Module):

    INPUT_LAYER_OUTPUT_SIZE = 512
    # This used to be 512 as well
    HIDDEN_LAYER_SIZE = 128
    OUTPUT_LAYER_INPUT_SIZE = HIDDEN_LAYER_SIZE // 2

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
        # pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print_with_timestamp(f"N parameters = {pytorch_total_params}, estimated amount of data needed = {pytorch_total_params * 10}")

    def forward(self, move_tensor_list: List[MoveTensor]):
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

        x = self.input_layer(x)
        x = F.leaky_relu(x)

        x = self.hidden_layer_1(x)
        x = F.leaky_relu(x)

        x = self.hidden_layer_2(x)
        x = F.leaky_relu(x)

        return self.output_layer(x)


def evaluate_model_with_metrics(model, test_loader, l):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for evaluation

    update_stopwatch = Stopwatch()
    update_stopwatch.start()

    print_with_timestamp("Starting model evaluation...")
    with torch.no_grad():  # Disable gradient computation
        for entry in test_loader:
            inputs = entry["data"]
            targets = entry["labels"]

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets.to(torch.float))  # Compute the loss
            # print_with_timestamp(inputs, targets, outputs, loss)
            total_loss += loss.item() * len(inputs)

            # Apply sigmoid to outputs and round to get binary predictions
            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == targets).sum().item()
            total += len(targets)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(torch.sigmoid(outputs).cpu().numpy())

            if update_stopwatch.has_minute_elapsed():
                print_with_timestamp(
                    f"Update: {(total / l) * 100:.2f}% done with evaluation"
                )
                update_stopwatch.reset()

    print_with_timestamp("Finished running model for validation. Now calculating...")
    average_loss = total_loss / total
    accuracy = correct / total
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # Calculate additional metrics
    precision = precision_score(all_targets, all_predictions.round())
    recall = recall_score(all_targets, all_predictions.round())
    f1 = f1_score(all_targets, all_predictions.round())
    roc_auc = roc_auc_score(all_targets, all_predictions)
    conf_matrix = confusion_matrix(all_targets, all_predictions.round())

    print_with_timestamp("Finalized evaluation calculations!")
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
    }


def print_pretty_table(title, d):
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.float_format = ".5"
    # Add rows to the table
    for key, value in d.items():
        table.add_row([key, value])

    print_with_timestamp(f"{title}:")
    print(table)


def main():
    # Create a DataLoader for the training data
    print_with_timestamp("DATASET LOADING")

    train_set, validation_set, test_set = create_lazy_loaders(
        "data3",
        # 10_000
        # 5_000_000
        36_000_000,
        # 10_000_000
    )

    # Initialize the model
    print_with_timestamp("NETWORK LOADING")
    model = CuttlefishNetwork()
    # FIXME: Try a bigger batch size
    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=False, collate_fn=custom_collate
    )
    validation_loader = DataLoader(
        validation_set, batch_size=64, shuffle=False, collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=False, collate_fn=custom_collate
    )
    print_with_timestamp("NETWORK READY")

    print_with_timestamp("OPTIMIZER LOADING")
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()  # Mean Squared Error Loss
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print_with_timestamp("OPTIMIZER READY")

    update_stopwatch = Stopwatch()
    update_stopwatch.start()

    # Training loop
    num_epochs = 50  # Number of epochs
    early_stopping_patience = 10
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        print_with_timestamp(f"STARTING EPOCH {epoch + 1}")
        model.train()  # Set the model to training mode
        train_set.shuffle()  # Shuffle the training set manually
        update_stopwatch.reset()  # Reset the timer to avoid pointless updates
        running_loss = 0.0
        running_loss_n = 0

        for entry in train_loader:
            inputs = entry["data"]
            targets = entry["labels"]

            # print_with_timestamp(f"{len(inputs)} inputs. {len(targets)} targets")

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute the loss

            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the weights

            if running_loss == 0:
                pytorch_total_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                print_with_timestamp(
                    f"N parameters = {pytorch_total_params:,}, estimated amount of data needed = {pytorch_total_params * 10:,}"
                )
            running_loss += loss.item() * len(inputs)
            running_loss_n += len(inputs)
            # print_with_timestamp(f"Loss: {loss.item():.4f}, Len {len(inputs)}, Running Loss: {running_loss:.4f}, Running Loss N: {running_loss_n:.4f}")

            if update_stopwatch.has_minute_elapsed():
                print_with_timestamp(
                    f"Update: {(running_loss_n / train_set.cap) * 100:.2f}% done with epoch {epoch + 1}"
                )
                update_stopwatch.reset()

        print_with_timestamp(f"Doing evaluation for epoch {epoch + 1}")
        model.eval()
        validation_metrics = evaluate_model_with_metrics(model, validation_loader, validation_set.cap)
        val_loss = validation_metrics["loss"]

        print_with_timestamp(f"Epoch [{epoch + 1}/{num_epochs}]")
        print_with_timestamp(f"Training Loss: {running_loss / running_loss_n:.6f}")
        print_pretty_table("Validation Data", validation_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print_with_timestamp("EARLY STOPPING TRIGGERED")
                break

    print_with_timestamp("DOING FINAL EVALUATION ON TEST SET")
    metrics = evaluate_model_with_metrics(model, test_loader, test_set.cap)
    print_with_timestamp(f"METRICS {metrics}")
    print_pretty_table("Test Metrics", metrics)


if __name__ == "__main__":
    main()
