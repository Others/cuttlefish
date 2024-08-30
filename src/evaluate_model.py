from datetime import timedelta

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader

import nets.net_128
import nets.net_512_128
import nets.net_512_256
import nets.net_2048
import nets.net_4096
import nets.net_quarter
from dataloader.new_loader import DirectFromPgnLoader
from utils.print_extra import print_pretty_table, print_with_timestamp
from utils.stopwatch import Stopwatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model_with_metrics(
    model, test_loader, duration, update_fun=print_with_timestamp
):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for evaluation

    update_stopwatch = Stopwatch()
    update_stopwatch.start()

    overall_stopwatch = Stopwatch()
    overall_stopwatch.start()

    update_fun("Starting model evaluation...")
    with torch.no_grad():  # Disable gradient computation
        for entry in test_loader:
            data = entry.data.to(device)
            labels = entry.labels.to(device)

            outputs = model(data)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            # print_with_timestamp(inputs, targets, outputs, loss)
            total_loss += loss.item() * len(outputs)

            # Apply sigmoid to outputs and round to get binary predictions
            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == labels).sum().item()
            total += len(data)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(torch.sigmoid(outputs).cpu().numpy())

            if update_stopwatch.has_minute_elapsed():
                update_fun(
                    f"Update: {overall_stopwatch.percentage_elapsed(duration)} done with evaluation"
                )
                update_stopwatch.reset()
                if overall_stopwatch.has_delta_elapsed(duration):
                    break

    update_fun("Finished running model for validation. Now calculating...")
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

    update_fun("Finalized evaluation calculations!")
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
    }


def main():
    data = [
        "data/lichess_db_standard_rated_2023-11.pgn.zst",
        "data/lichess_db_standard_rated_2023-12.pgn.zst",
    ]

    models = [
        (
            "best_model_826.pth",
            nets.net_4096.CuttlefishNetwork(),
            nets.net_4096.Collator(),
        ),
        (
            "best_model_825.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_824.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_822.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_821.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_820.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_819.pth",
            nets.net_quarter.CuttlefishNetwork(),
            nets.net_quarter.Collator(),
        ),
        (
            "best_model_818.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_816.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_814.pth",
            nets.net_2048.CuttlefishNetwork(),
            nets.net_2048.Collator(),
        ),
        (
            "best_model_813.pth",
            nets.net_512_256.CuttlefishNetwork(),
            nets.net_512_256.Collator(),
        ),
        (
            "best_model_812.pth",
            nets.net_512_256.CuttlefishNetwork(),
            nets.net_512_256.Collator(),
        ),
        (
            "best_model_731.pth",
            nets.net_512_128.CuttlefishNetwork(),
            nets.net_512_128.Collator(),
        ),
        (
            "best_model_128.pth",
            nets.net_128.CuttlefishNetwork(),
            nets.net_128.Collator(),
        ),
    ]

    duration = timedelta(minutes=5)
    print_with_timestamp(f"Estimated time to completion: {duration * 2 * len(models)}")

    results = []

    for file, net, collator in models:
        print_with_timestamp(f"Evaluating model {file}")

        evaluation_loader = DataLoader(
            DirectFromPgnLoader(data),
            num_workers=len(data),
            batch_size=64,
            shuffle=False,
            collate_fn=collator.custom_collate,
        )

        balanced_evaluation_loader = DataLoader(
            DirectFromPgnLoader(data, game_length_min=100),
            num_workers=len(data),
            batch_size=64,
            shuffle=False,
            collate_fn=collator.custom_collate,
        )

        net.load_state_dict(torch.load(file, weights_only=True))
        net.to(device)

        evaluation = evaluate_model_with_metrics(net, evaluation_loader, duration)
        print_pretty_table("Regular Evaluation", evaluation)

        evaluation_balanced = evaluate_model_with_metrics(
            net, balanced_evaluation_loader, duration
        )
        print_pretty_table("Balanced Evaluation", evaluation_balanced)

        results.append(
            (
                file,
                evaluation["loss"],
                evaluation["accuracy"],
                evaluation["f1_score"],
                evaluation["roc_auc"],
                evaluation_balanced["loss"],
                evaluation_balanced["accuracy"],
                evaluation_balanced["f1_score"],
                evaluation_balanced["roc_auc"],
            )
        )

    table = PrettyTable()
    table.field_names = [
        "Model",
        "Loss",
        "Accuracy",
        "F1 Score",
        "ROC AUC",
        "Balanced Loss",
        "Balanced Accuracy",
        "Balanced F1 Score",
        "Balanced ROC AUC",
    ]
    table.float_format = ".5"
    for r in results:
        table.add_row(r)
    print_with_timestamp("Model Evaluation Table")
    print(table)


if __name__ == "__main__":
    main()
