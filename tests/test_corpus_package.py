import sys
import unittest
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_corpus_subset import select


class PortableSelectionTests(unittest.TestCase):
    def test_gold_minimum_adjusts_quotas_and_is_order_independent(self):
        index = {f'{s}{i}': f'{s}/{i}.json' for s, n in [('large', 90), ('small', 10)] for i in range(n)}
        required = {'small0', 'small1', 'small2'}
        selected, quotas, base = select(index, 10, required, 42)
        self.assertEqual(base, {'large': 9, 'small': 1})
        self.assertEqual(quotas, {'large': 7, 'small': 3})
        self.assertTrue(required <= selected)
        self.assertEqual(len(selected), 10)
        self.assertEqual(Counter(index[d].split('/')[0] for d in selected), quotas)
        self.assertEqual(select(dict(reversed(list(index.items()))), 10, required, 42)[0], selected)

    def test_impossible_selection_fails(self):
        index = {'a': 'slack/a', 'b': 'jira/b'}
        for count, required in [(1, {'a', 'b'}), (3, set()), (1, {'missing'})]:
            with self.assertRaises(ValueError):
                select(index, count, required, 42)


if __name__ == '__main__':
    unittest.main()
