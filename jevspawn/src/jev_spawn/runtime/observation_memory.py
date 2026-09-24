import json


class ObservationMemory:
    def __init__(self, initial, actions, serialization):
        self.actions, self.serialization = list(actions), serialization
        self.observations, self.indices = [], {}
        self.transitions, self.events, self.edge_indices = [], [], {}
        self.current = self.identify(initial)

    def identify(self, observation):
        key = json.dumps(observation, **self.serialization)
        if key not in self.indices:
            self.indices[key] = len(self.observations)
            self.observations.append(observation)
        return self.indices[key]

    def record(self, action, feedback):
        target = self.identify(feedback)
        edge = (self.current, action, target)
        if edge not in self.edge_indices:
            self.edge_indices[edge] = len(self.transitions)
            self.transitions.append({'source': self.current, 'action': action, 'target': target})
        self.events.append(self.edge_indices[edge])
        self.current = target

    def view(self):
        used = {(edge['source'], edge['action']) for edge in self.transitions}
        return {'current_observation': self.current, 'observations': self.observations,
                'transitions': self.transitions, 'event_sequence': self.events,
                'unexecuted_actions': [[action for action in self.actions if (index, action) not in used]
                                      for index in range(len(self.observations))]}
