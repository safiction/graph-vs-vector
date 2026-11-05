import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coverage import score_coverage


class CoverageTests(unittest.TestCase):
    def test_hand_worked_example(self):
        gold = ['dsid_e33c122264214846a8b7df7701d31482', 'dsid_409618a3958944f9b78362ed7e01bb33',
                'dsid_a5a5235d973e421c870e11ceea4887ed', 'dsid_4b2ac6984bdf4737a18520e158b04453']
        result = score_coverage('qst_0344', gold, [gold[0], gold[0], gold[1], 'dsid_other'], {})
        self.assertEqual((result['coverage'], result['full_coverage'], result['source_count']), (0.5, False, 3))

    def test_empty_context_and_complete_context(self):
        self.assertEqual(score_coverage('q', ['a', 'b'], [], {})['coverage'], 0)
        self.assertTrue(score_coverage('q', ['a', 'b'], ['b', 'a'], {})['full_coverage'])

    def test_missing_gold_and_explicit_exclusion_are_na(self):
        self.assertIsNone(score_coverage('q', [], ['a'], {})['coverage'])
        self.assertIsNone(score_coverage('qst_0413', ['a'], ['a'], {'qst_0413': {'reason': 'collision'}})['coverage'])

    def test_missing_provenance_is_not_silently_empty(self):
        with self.assertRaises(ValueError):
            score_coverage('q', ['a'], None, {})


if __name__ == '__main__':
    unittest.main()
