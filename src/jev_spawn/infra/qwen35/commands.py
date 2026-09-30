import torch.distributed as dist


class ParallelCommands:
    def __init__(self, control_group, tensor_group, settings):
        self.control_group, self.tensor_group = control_group, tensor_group
        self.settings = settings
        self.leader_rank = settings['leader_rank']
        self.is_leader = dist.get_rank() == self.leader_rank
        self.handlers = {}

    def register(self, name, handler):
        assert name not in self.handlers
        self.handlers[name] = handler

    def leader_value(self, value):
        values = [value]
        dist.broadcast_object_list(values, src=self.leader_rank, group=self.control_group)
        return values.pop()

    def broadcast_tensor(self, tensor):
        dist.broadcast(tensor, src=self.leader_rank, group=self.tensor_group)
        return tensor

    def call(self, name, payload):
        assert self.is_leader
        self.leader_value((self.settings['operations']['call'], name, payload))
        return self.handlers[name](payload)

    def _receive(self, terminal):
        while True:
            operation, name, payload = self.leader_value(None)
            if operation == terminal:
                return
            assert operation == self.settings['operations']['call']
            self.handlers[name](payload)

    def serve(self):
        assert not self.is_leader
        self._receive(self.settings['operations']['finish'])

    def finish(self):
        assert self.is_leader
        self.leader_value((self.settings['operations']['finish'], None, None))

    def window(self, callback):
        terminal = self.settings['operations']['return']
        if self.is_leader:
            try:
                return callback()
            finally:
                self.leader_value((terminal, None, None))
        self._receive(terminal)
