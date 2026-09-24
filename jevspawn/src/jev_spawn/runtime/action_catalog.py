from copy import deepcopy
import json


def encode_catalog(candidates, settings):
    templates, groups, literals = {}, {}, []
    metadata_key = settings['metadata_key']
    for index, candidate in enumerate(candidates):
        if metadata_key not in candidate:
            literals.append([index, deepcopy(candidate)])
            continue
        descriptor = candidate[metadata_key]
        name, template = descriptor['name'], descriptor['template']
        if name in templates:
            assert templates[name] == template
        templates[name] = template
        parameters = {key: json.loads(value) if key in settings['json_fields'] else value
                      for key, value in descriptor['parameters'].items()}
        row_fields = [key for key in settings['row_fields'] if key in parameters]
        common = {key: value for key, value in parameters.items() if key not in row_fields}
        key = json.dumps([name, common, row_fields], **settings['group_serialization'])
        if key not in groups:
            groups[key] = {'template': name, 'parameters': common,
                           'columns': [*settings['columns'], *row_fields], 'rows': []}
        groups[key]['rows'].append([index, candidate['id'], *[parameters[field] for field in row_fields]])
    for group in groups.values():
        group['path_prefixes'] = {}
        for field in settings['path_fields']:
            if field in group['columns']:
                position = group['columns'].index(field)
                paths = [row[position] for row in group['rows']]
                prefix = []
                for column in zip(*paths):
                    first, *others = column
                    if not all(type(value) is type(first) and value == first for value in others):
                        break
                    prefix.append(first)
                group['path_prefixes'][field] = prefix
                for row in group['rows']:
                    row[position] = row[position][len(prefix):]
    return {'templates': templates, 'groups': list(groups.values()), 'literal': literals}


def decode_catalog(encoded, settings):
    indexed = dict(deepcopy(encoded['literal']))
    for group in encoded['groups']:
        name = group['template']
        template = encoded['templates'][name]
        for row in group['rows']:
            values = dict(zip(group['columns'], row, strict=True))
            values.update({field: [*prefix, *values[field]] for field, prefix in group['path_prefixes'].items()})
            parameters = {**group['parameters'], **{key: value for key, value in values.items()
                                                   if key not in settings['columns']}}
            formatted = {key: json.dumps(value, **settings['serialization']) if key in settings['json_fields'] else value
                         for key, value in parameters.items()}
            index, identity = [values[key] for key in settings['columns']]
            assert index not in indexed
            indexed[index] = {'id': identity, 'description': template.format(**formatted),
                settings['metadata_key']: {'name': name, 'template': template, 'parameters': formatted}}
    return [indexed[index] for index in sorted(indexed)]
