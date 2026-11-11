import sys
import unittest
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from index_graph_subset import select_documents


class SubsetSelectionTests(unittest.TestCase):
    def test_selection_is_proportional_deterministic_and_includes_required_documents(self):
        records = [{'source_document_id': f'{source}_{i}', 'path': f'{source}/{i}.json'}
                   for source, n in [('slack', 6), ('gmail', 3), ('jira', 1)] for i in range(n)]
        required = {'slack_5', 'gmail_2'}
        selected, quotas = select_documents(records, 5, required, seed=42)
        again, _ = select_documents(list(reversed(records)), 5, required, seed=42)
        self.assertEqual(selected, again)
        self.assertEqual(len(selected), 5)
        self.assertTrue(required <= selected)
        actual = Counter(r['path'].split('/')[0] for r in records if r['source_document_id'] in selected)
        self.assertEqual({source: actual[source] for source in quotas}, quotas)
        self.assertEqual(quotas, {'slack': 3, 'gmail': 2, 'jira': 0})

    def test_unknown_required_documents_and_duplicate_ids_are_rejected(self):
        rows = [{'source_document_id': 'a', 'path': 'slack/a.json'}]
        with self.assertRaises(ValueError):
            select_documents(rows, 1, {'missing'})
        with self.assertRaises(ValueError):
            select_documents(rows + rows, 1)


if __name__ == '__main__':
    unittest.main()
