import time
import logging

class TimedSection(object):
    def __init__(self, label = "Section completed in %.3f seconds", logger = "maskrcnn_benchmark.timed_section"):
        self.logger = logging.getLogger(logger)
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        self.logger.info(self.label % (time.perf_counter() - self.start))

