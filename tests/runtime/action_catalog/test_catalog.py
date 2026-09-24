from copy import deepcopy
import json
from pathlib import Path
import unittest

from jev_spawn.runtime.action_catalog import decode_catalog, encode_catalog


CONFIG = json.loads(Path(__file__).with_name('fixtures.json').read_text())
SETTINGS = json.loads(Path(CONFIG['settings']).read_text())


class CatalogTests(unittest.TestCase):
    def candidates(self):
        values = [deepcopy(CONFIG['literal'])]
        for row in CONFIG['rows']:
            parameters = {key: json.dumps(value, **SETTINGS['serialization']) if key in SETTINGS['json_fields'] else value
                          for key, value in row.items() if key != 'id'}
            values.append({'id': row['id'], 'description': CONFIG['template'].format(**parameters),
                SETTINGS['metadata_key']: {'name': CONFIG['template_name'], 'template': CONFIG['template'],
                                            'parameters': parameters}})
        return values

    def test_exact_order_values_nonnumeric_addresses_and_no_mutation(self):
        original = self.candidates()
        saved = deepcopy(original)
        encoded = encode_catalog(original, SETTINGS)
        decoded = decode_catalog(encoded, SETTINGS)
        self.assertEqual(json.dumps(decoded, sort_keys=True), json.dumps(original, sort_keys=True))
        self.assertEqual(original, saved)
        self.assertEqual([candidate['id'] for candidate in decoded], [candidate['id'] for candidate in original])

    def test_literal_only_catalog_preserves_complete_values(self):
        values = [CONFIG['literal']]
        self.assertEqual(decode_catalog(encode_catalog(values, SETTINGS), SETTINGS), values)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CatalogTests))
    Path(CONFIG['result']).write_text(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun}, indent=2) + '\n')
    raise SystemExit(not result.wasSuccessful())
