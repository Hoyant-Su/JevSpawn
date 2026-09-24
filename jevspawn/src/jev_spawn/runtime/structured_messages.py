from copy import deepcopy


def record_table(records, columns):
    return {'columns': list(columns),
            'rows': [[deepcopy(record[column]) for column in columns] for record in records]}


def judgment_table(judgments, schema):
    records = [{**judgment, 'bindings': [[deepcopy(binding[column]) for column in schema['binding_columns']]
                                       for binding in judgment['bindings']]}
               for judgment in judgments]
    return {**record_table(records, schema['judgment_columns']),
            'binding_columns': list(schema['binding_columns'])}
