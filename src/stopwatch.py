from datetime import datetime, timedelta


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.reset_count = 0

    def start(self):
        self.start_time = datetime.now()

    def has_minute_elapsed(self):
        if self.start_time is None:
            return False
        return datetime.now() - self.start_time >= timedelta(minutes=1)

    def reset(self):
        self.start_time = datetime.now()
        self.reset_count += 1
