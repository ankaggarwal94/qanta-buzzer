"""Unit guard fixtures, explicitly not model generations or experiment results."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import tempfile
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_acl_expansion as p


def trajectories():
    def row(qid, idx, fraction, confidence, grade):
        return {'qid': qid, 'job_id': f'{qid}-{idx}', 'prefix_id': f'p{idx}',
                'fraction': fraction, 'confidence': confidence, 'status': 'answer',
                'grade': grade, 'correct': None if grade=='needs_review' else grade=='accepted'}
    return {'q1': [row('q1',1,.1,.9,'needs_review'),row('q1',2,1.,.95,'accepted')],
            'q2': [row('q2',1,.1,.5,'rejected'),row('q2',2,1.,.9,'accepted')]}


class AnalysisGuardTests(unittest.TestCase):
    def test_fixed_policy_keeps_unresolved_in_risk_interval(self):
        result = p.fixed_policy_bounds(trajectories(), {'x':[0,1],'y':[0,1]}, {'mode':'threshold','value':.9})
        assert result['n_committed']==2
        assert result['n_unresolved_commits']==1
        assert result['n_known_incorrect_commits']==0
        assert result['risk_identification_interval']==[0,.5]
        assert result['mean_commitment_fraction']==.55
        # Earliest answer is selected, not the later known-correct answer.
        assert result['commitments'][0]['prefix_id']=='p1'


    def test_never_retains_undefined_conditional_metrics(self):
        result = p.fixed_policy_bounds(trajectories(), {'x':[0,1],'y':[0,1]}, {'mode':'never','value':None})
        assert result['coverage']==0
        assert result['risk_identification_interval'] is None
        assert result['mean_commitment_fraction'] is None


    def test_resolving_grade_changes_risk_but_not_fixed_commitments(self):
        before = trajectories(); after = copy.deepcopy(before)
        after['q1'][0].update(grade='rejected',correct=False)
        c={'x':[0,1],'y':[0,1]}; t={'mode':'threshold','value':.9}
        a=p.fixed_policy_bounds(before,c,t);b=p.fixed_policy_bounds(after,c,t)
        assert b['risk_identification_interval']==[.5,.5]
        assert a['n_committed']==b['n_committed']
        assert a['mean_commitment_fraction']==b['mean_commitment_fraction']


    def test_accuracy_bounds_weight_questions_and_keep_abstentions(self):
        rows=[{'qid':'a','grade':'accepted'},{'qid':'a','grade':'needs_review'},
              {'qid':'b','grade':'abstain'}]
        assert p.accuracy_bounds(rows)['accuracy_identification_interval']==[.25,.5]


    def test_contract_rejects_mutated_frozen_input_before_dataset_read(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        tmp_path = Path(temp.name)
        frozen=tmp_path/'frozen';frozen.mkdir();path=frozen/'manifest.json';path.write_text('{}')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        contract=tmp_path/'contract.json'
        contract.write_text(json.dumps({'frozen_inputs_sha256':{'manifest.json':digest},'code_inputs_sha256':{}}))
        path.write_text('{"mutated":true}')
        with self.assertRaisesRegex(ValueError, 'frozen analysis input changed: manifest.json'):
            p.verify_contract(frozen,contract)


    def test_unknown_grades_are_not_modified_by_primary_bounds(self):
        rows=[{'qid':'a','grade':'needs_review','correct':None}]
        before=copy.deepcopy(rows)
        assert p.accuracy_bounds(rows)['accuracy_identification_interval']==[0,1]
        assert rows==before

if __name__ == '__main__':
    unittest.main()
