from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader.new_loader import CircularDataset, DirectFromPgnLoader
from evaluate_model import evaluate_model_with_metrics
from nets.net_2048_focus import Collator, CuttlefishNetwork
from utils.print_extra import print_pretty_table, print_with_timestamp
from utils.stopwatch import Stopwatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_with_data(data, labels, model, optimizer, criterion):
    # print_with_timestamp(f"{len(inputs)} inputs. {len(targets)} targets")

    optimizer.zero_grad()  # Zero the parameter gradients

    outputs = model(data)  # Forward pass
    loss = criterion(outputs, labels)  # Compute the loss

    loss.backward()  # Backward pass
    optimizer.step()  # Optimize the weights

    return loss


def run_epoch(
    epoch,
    model,
    train_loader,
    train_time,
    optimizer,
    criterion,
    update_stopwatch,
    overall_stopwatch,
):
    model.train()  # Set the model to training mode
    model.to(device)
    update_stopwatch.reset()  # Reset the timer to avoid pointless updates
    overall_stopwatch.reset()
    running_loss = 0.0
    running_loss_n = 0

    for entry in train_loader:
        data = entry.data.to(device)
        labels = entry.labels.to(device)

        loss = train_with_data(data, labels, model, optimizer, criterion)

        if epoch > 1 and running_loss == 0:
            pytorch_total_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print_with_timestamp(
                f"N parameters = {pytorch_total_params:,}, estimated amount of data needed = {pytorch_total_params * 10:,}"
            )

        running_loss += loss.item() * len(labels)
        running_loss_n += len(labels)

        if update_stopwatch.has_minute_elapsed():
            print_with_timestamp(
                f"Update: {overall_stopwatch.percentage_elapsed(train_time)} done with epoch {epoch + 1}"
            )
            update_stopwatch.reset()

            if overall_stopwatch.has_delta_elapsed(train_time):
                break

    return running_loss, running_loss_n


def main():
    # Create a DataLoader for the training data
    print_with_timestamp("Dataset loading...")
    batch_size = 64

    # Setup the collator
    collator = Collator()

    # FIXME: Try a bigger batch size
    train_circle = CircularDataset(
        DirectFromPgnLoader(
            [
                "data/lichess_db_standard_rated_2024-05.pgn.zst",
                "data/lichess_db_standard_rated_2024-04.pgn.zst",
                "data/lichess_db_standard_rated_2024-03.pgn.zst",
            ]
        )
    )
    train_loader = DataLoader(
        train_circle,
        num_workers=2,
        batch_size=batch_size,
        prefetch_factor=3,
        shuffle=False,
        collate_fn=collator.custom_collate,
    )
    train_time = timedelta(minutes=30)

    validation_loader = DataLoader(
        DirectFromPgnLoader(
            [
                "data/lichess_db_standard_rated_2024-06.pgn.zst",
                "data/lichess_db_standard_rated_2024-02.pgn.zst",
            ]
        ),
        num_workers=2,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator.custom_collate,
    )
    validation_time = train_time * 0.5

    test_loader = DataLoader(
        DirectFromPgnLoader(
            [
                "data/lichess_db_standard_rated_2024-07.pgn.zst",
                "data/lichess_db_standard_rated_2024-01.pgn.zst",
            ]
        ),
        num_workers=2,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator.custom_collate,
    )
    test_time = train_time * 2

    # Initialize the model
    print_with_timestamp("Network loading...")
    model = CuttlefishNetwork()
    print_with_timestamp(f"Device set to: {device}")

    print_with_timestamp("Optimizer loading...")
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()  # Mean Squared Error Loss
    # Define the optimizer (this lr is the default BTW)
    optimizer = optim.Adamax(model.parameters(), lr=2e-3, weight_decay=1e-5)

    print_with_timestamp("Beginning training...")
    overall_stopwatch = Stopwatch()
    overall_stopwatch.start()
    update_stopwatch = Stopwatch()
    update_stopwatch.start()

    # Training loop
    num_epochs = 250  # Number of epochs
    early_stopping_patience = num_epochs / 2
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):

        print_with_timestamp(f"Starting training for Epoch [{epoch + 1}/{num_epochs}]")

        (loss, loss_n) = run_epoch(
            epoch,
            model,
            train_loader,
            train_time,
            optimizer,
            criterion,
            update_stopwatch,
            overall_stopwatch,
        )

        print_with_timestamp(f"Trained on {loss_n:,} samples")
        print_with_timestamp(f"Doing evaluation for Epoch [{epoch + 1}/{num_epochs}]")
        model.eval()
        validation_metrics = evaluate_model_with_metrics(
            model, validation_loader, validation_time
        )
        val_loss = validation_metrics["loss"]

        print_with_timestamp(f"Training Loss: {loss / loss_n:.6f}")
        print_pretty_table("Validation Data", validation_metrics)

        print_with_timestamp(f"Finished Epoch [{epoch + 1}/{num_epochs}]")

        if val_loss < best_val_loss:
            print_with_timestamp(
                f"New best model! Validation loss of {val_loss:.4f} beat the record of {best_val_loss:.4f}"
            )
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print_with_timestamp(
                    f"Stopping early due to failure to learn for {early_stopping_patience} epochs..."
                )
                break
        print()

    print_with_timestamp("Doing final evaluation on test set...")
    metrics = evaluate_model_with_metrics(model, test_loader, test_time)
    print_with_timestamp(f"Raw metrics {metrics}")
    print_pretty_table("Test Metrics", metrics)


if __name__ == "__main__":
    main()
