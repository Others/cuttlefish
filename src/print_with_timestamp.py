from datetime import datetime


# TODO: Switch to using a real logging framework
def print_with_timestamp(s):
    print(datetime.now(), s)
