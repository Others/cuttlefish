from datetime import datetime, timedelta


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.reset_count = 0

    def start(self):
        self.start_time = datetime.now()

    def percentage_elapsed(self, delta):
        fraction = 0.0
        if self.start_time is not None:
            fraction = (
                datetime.now() - self.start_time
            ).total_seconds() / delta.total_seconds()
            fraction = min(fraction, 1.0)
        return f"{fraction * 100:.2f}%"

    def has_delta_elapsed(self, delta):
        if self.start_time is None:
            return False
        return datetime.now() - self.start_time >= delta

    def has_minute_elapsed(self):
        return self.has_delta_elapsed(timedelta(minutes=1))

    def reset(self):
        self.start_time = datetime.now()
        self.reset_count += 1
