from torch.utils.tensorboard import SummaryWriter

SUM_FREQ = 100


class Logger:
    def __init__(self, name, scheduler):
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.name = name
        self.scheduler = scheduler

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter("runs/%s" % self.name)  # noqa: UP031 - Keeps printf conversions that depend on argument types.
            print(list(self.running_loss))

        lr = self.scheduler.get_lr().pop()
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in self.running_loss.keys()]  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
        training_str = f"[{self.total_steps + 1:6d}, {lr:10.7f}] "
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics):
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
