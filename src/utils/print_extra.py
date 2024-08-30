from datetime import datetime

from prettytable import PrettyTable


# TODO: Switch to using a real logging framework
def print_with_timestamp(s):
    print(datetime.now(), s)


def print_pretty_table(title, d):
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.float_format = ".5"
    # Add rows to the table
    for key, value in d.items():
        table.add_row([key, value])

    print_with_timestamp(f"{title}:")
    print(table)
