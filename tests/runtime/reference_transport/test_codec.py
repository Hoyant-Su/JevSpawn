from copy import deepcopy
import json
from pathlib import Path
import unittest

from jev_spawn.runtime.reference_transport import pack_references, unpack_references


class ReferenceTransportTest(unittest.TestCase):
    def test_exact_types_collisions_order_and_backward_references(self):
        settings = json.loads(Path('configs/jevspawn/reference_transport.json').read_text())
        fixtures = json.loads(Path('tests/runtime/reference_transport/fixtures.json').read_text())
        ref, escaped = settings['reference_key'], settings['object_key']

        def check(item, available):
            if isinstance(item, dict):
                if ref in item:
                    self.assertEqual(list(item), [ref])
                    self.assertIn(item[ref], range(available))
                elif escaped in item:
                    self.assertEqual(list(item), [escaped])
                    for _, value in item[escaped]:
                        check(value, available)
                else:
                    for value in item.values():
                        check(value, available)
            elif isinstance(item, list):
                for value in item:
                    check(value, available)

        for original in fixtures:
            snapshot = deepcopy(original)
            packed = pack_references(original, settings)
            self.assertEqual(json.dumps(unpack_references(packed, settings), **settings['serialization']),
                             json.dumps(original, **settings['serialization']))
            self.assertEqual(original, snapshot)
            entries = packed[settings['dictionary_key']]
            for index, entry in enumerate(entries):
                check(entry, index)
            check(packed[settings['root_key']], len(entries))


if __name__ == '__main__':
    unittest.main()
