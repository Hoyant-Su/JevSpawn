from jev_spawn.service.values import ValueRequest


def readout(service, fields, task_id, settings):
    return service.enqueue_decisions(fields, task_id=task_id, request_type=ValueRequest)
