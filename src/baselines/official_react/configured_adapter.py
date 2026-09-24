import ast
import json
from pathlib import Path
import re

def thought_action_parts(text, index):
    marker = re.search(rf'(?:^|\n)Action {index}:\s*', text.strip())
    if marker is None:
        raise ValueError('The model response contains no action for the current turn.')
    thought = text.strip()[:marker.start()]
    action = text.strip()[marker.end():].strip()
    tool = re.fullmatch(r'[A-Za-z_]+\[(.*)\]', action, flags=re.S)
    if tool is None:
        raise ValueError('The model action does not match the tool transport format.')
    json.loads(tool.group(1))
    return thought, action


def original_function(notebook, namespace, *, episode_action_limit, original_episode_range):
    cells = json.loads(Path(notebook).read_text())['cells']
    source = next(''.join(cell['source']) for cell in cells
                  if cell['cell_type'] == 'code' and '\ndef webthink(' in ''.join(cell['source']))
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'webthink')
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == 'range']
    assert len(loops) == 1
    episode = loops[0]
    assert [ast.literal_eval(argument) for argument in episode.iter.args] == original_episode_range
    assert isinstance(episode_action_limit, int) and episode_action_limit > 0
    episode.iter.args[1] = ast.Constant(value=original_episode_range[0] + episode_action_limit)
    for node in ast.walk(function):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                and [target.id for target in node.targets[0].elts] == ['thought', 'action']):
            node.value = ast.Call(func=ast.Name(id='thought_action_parts', ctx=ast.Load()),
                                  args=[ast.Name(id='thought_action', ctx=ast.Load()),
                                        ast.Name(id='i', ctx=ast.Load())], keywords=[])
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == 'action' and isinstance(node.slice, ast.Constant)
                and node.slice.value == 0):
            node.slice = ast.Slice(upper=ast.Constant(value=1))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace['thought_action_parts'] = thought_action_parts
    exec(compile(module, str(notebook), 'exec'), namespace)
    return namespace['webthink']
