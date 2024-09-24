from collections import deque


class LossMonitor:
    def __init__(self, *losses, window: int):
        self.window = window
        self.loss_queues = {}
        self.loss_queue_sums = {}
        for name in losses:
            self.loss_queues[name] = deque()
            self.loss_queue_sums[name] = 0

    def update(self, **loss_values):
        assert len(loss_values) == len(self.loss_queues)
        for name in loss_values:
            if len(self.loss_queues[name]) == self.window:
                self.loss_queue_sums[name] -= self.loss_queues[name].popleft()
            self.loss_queues[name].append(loss_values[name])
            self.loss_queue_sums[name] += loss_values[name]
        return {name: self.loss_queue_sums[name] / len(self.loss_queues[name]) for name in self.loss_queue_sums}
