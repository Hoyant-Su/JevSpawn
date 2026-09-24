from baselines.common.context_window import truncate_prompt
from baselines.latentmas.common_service import LatentMASService, role_messages
from baselines.latentmas.role_window import role_budgets
from jev_spawn.infra.prompts import load_prompt


class WindowedLatentMASService(LatentMASService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts, input_policy):
        self.input_policy = input_policy
        self.input_notice = load_prompt(self.input_policy['prompt'])
        self.input_truncations = []
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata['input_policy'] = self.input_policy

    def _validate_inputs(self, batch):
        valid, rendered_rows = [], []
        for request in batch:
            try:
                self.deadlines.remaining(request.task_id)
            except TimeoutError as error:
                request.future.set_exception(error)
                continue
            request.role_messages = role_messages(request.messages, self.prompts, self.shared.model.path)
            rendered = [self.wrapper.render_chat(messages) for messages in request.role_messages]
            request.role_ids = self.backend.tokenizer(rendered, add_special_tokens=False,
                                                       truncation=False)['input_ids']
            valid.append(request)
            rendered_rows.append(rendered)
        if not valid:
            return valid
        agents = self.agents
        latent_tokens = sum(agent.role != 'judger' for agent in agents) * self.settings['latent_steps']
        capacity = self.shared.model.max_input_tokens - latent_tokens
        demands = [max(len(request.role_ids[index]) for request in valid) for index in range(len(agents))]
        budgets = role_budgets(demands, capacity, self.settings['role_window'])
        for request, rendered in zip(valid, rendered_rows, strict=True):
            for role, (text, tokens, budget) in enumerate(zip(rendered, request.role_ids, budgets, strict=True)):
                effective, ids, metadata = truncate_prompt(self.backend.tokenizer, text, tokens, budget,
                                                         self.input_policy, self.input_notice)
                request.role_ids[role] = ids
                if metadata['omitted_tokens']:
                    self.input_truncations.append({'task_id': request.task_id, 'role': agents[role].role,
                        'role_index': role, 'role_budgets': budgets, 'original_role_padded_widths': demands,
                        'latent_tokens_reserved': latent_tokens, 'combined_input_limit': self.shared.model.max_input_tokens,
                        'effective_rendered': effective, 'effective_input_ids': ids, **metadata})
            request.input_ids = request.role_ids[0]
        physical = sum(max(len(request.role_ids[index]) for request in valid) for index in range(len(agents)))
        assert physical + latent_tokens <= self.shared.model.max_input_tokens
        return valid
