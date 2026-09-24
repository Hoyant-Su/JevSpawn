import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from catalog_frontier import admission, save
from jev_spawn.algo.structured import common_prefix


def prepare(settings):
    source=json.loads(Path(settings['feasibility_output']).read_text())
    templates=json.loads(Path(settings['templates']).read_text())
    shared=SharedConfig.load(settings['shared_config'])
    backend=SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(shared.model.path),
        answer_labels=source['answer_labels'],config={'max_input_tokens':shared.model.max_input_tokens})
    plans={}
    for identity in dict.fromkeys(group['frontier_id']for group in source['groups']):
        groups=[group for group in source['groups']if group['frontier_id']==identity]
        admissions=[admission(backend,group,'catalog',templates,settings)for group in groups]
        lengths={row['base_length']for row in admissions}
        assert len(lengths)==1
        original=next(iter(lengths))
        sequences=[list(prompt.tokens)for row in admissions for prompt in row['admitted']]
        shared_length=common_prefix(sequences)
        corrected=shared_length if len(groups)>1 else original
        assert original<=corrected
        assert all(corrected<group['layouts']['catalog']['prefix_tokens']for group in groups)
        prefix=sequences[0][:corrected]
        assert all(tokens[:corrected]==prefix for tokens in sequences)
        savings=(corrected-original)*(len(groups)-1)
        plans[identity]={'original_base_length':original,'corrected_base_length':corrected,
            'complete_group_prefix_lengths':[group['layouts']['catalog']['prefix_tokens']for group in groups],
            'group_count':len(groups),'field_ids':[field['id']for group in groups for field in group['fields']],
            'expected_cold_computed_token_savings':savings,
            'same_exact_tokens_all_fields':True,
            'rule':'Use deepest exact common token prefix across multiple groups of one declared frontier. A single group keeps existing base prefix; no additional cross-field regrouping.'}
    output=Path(settings['prefix_plan_output']);output.parent.mkdir(parents=True,exist_ok=True)
    result={'settings':settings,'frontiers':plans,'all_original_fields':sum(len(row['field_ids'])for row in plans.values()),
        'expected_total_cold_token_savings':sum(row['expected_cold_computed_token_savings']for row in plans.values())}
    save(output,result)
    print(json.dumps({key:value for key,value in result.items()if key!='settings'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    prepare(json.loads(args.config.read_text()))
