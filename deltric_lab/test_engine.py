"""Meaningful trace checks: production parity, timing, budgets, and hop arithmetic."""
import unittest
import numpy as np
import engine


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        # Two fixed sources contest a three-point chain; one remote point is over budget.
        self.data = dict(
            base=np.array([0, 0, -1, -1, -1, 1, 1, -1]),
            y=np.array([0, 0, -1, 0, -1, 1, 1, -1]),
            X=np.arange(16, dtype=float).reshape(8, 2),
            edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [4, 7]]),
            lengths=np.array([1., 1., 1., 1., 1., 1., 40.]),
            core=np.array([1., 1., .5, 2., 1., 1., 1., 1.]),
        )
        self.settings = dict(reach=4., penalty=.5, density=.5)
        self.meta = dict(settings=self.settings)

    def test_matches_production_and_reports_competition(self):
        run = engine.replay(self.data, self.settings)
        expected, info = engine.gate(self.data['base'], self.data['edges'], self.data['lengths'], 8,
            .5, 4., core_distance=self.data['core'], density_power=.5, density_clip=False)
        np.testing.assert_array_equal(run['final'], expected)
        self.assertEqual(len(run['claims']), info['n_claims'])
        self.assertTrue(any(e['status'] == 'owned' for e in run['_events']))
        self.assertTrue(any(e['node'] == 7 and e['status'] == 'budget' for e in run['_events']))

    def test_costs_and_temporal_inspection(self):
        run = engine.replay(self.data, self.settings)
        for i, claim in enumerate(run['claims']):
            node = int(claim[0])
            before = engine.point_evidence(run, self.meta, node, i)
            after = engine.point_evidence(run, self.meta, node, i + 1)
            self.assertEqual(before['visible'], -1)
            self.assertEqual(after['visible'], int(claim[1]))
            self.assertAlmostEqual(sum(h['hop_cost'] for h in after['hops']), claim[2])
            self.assertFalse(next(e for e in before['proposals'] if e['status'] == 'accepted')['visible'])
            self.assertTrue(next(e for e in after['proposals'] if e['status'] == 'accepted')['visible'])

    def test_zero_reach_and_density_floor(self):
        zero = engine.replay(self.data, {**self.settings, 'reach': 0.})
        self.assertEqual(len(zero['claims']), 0)
        np.testing.assert_array_equal(zero['final'], self.data['base'])
        floor_settings = {**self.settings, 'density_clip': True}
        floor = engine.replay(self.data, floor_settings)
        point = engine.point_evidence(floor, dict(settings=floor_settings), 2, len(floor['claims']))
        self.assertAlmostEqual(point['hops'][0]['hop_cost'], 1.)
        self.assertAlmostEqual(point['hops'][0]['cumulative'], 1.)

    def test_normal_claim_compares_destination_majority(self):
        run = engine.replay(self.data, self.settings)
        point = engine.point_evidence(run, self.meta, 3, len(run['claims']))
        self.assertIn("truth 0, matching this point", point['explanation'])
        run['y'] = run['y'].copy()
        run['y'][3] = 1
        point = engine.point_evidence(run, self.meta, 3, len(run['claims']))
        self.assertIn("truth 0, not 1", point['explanation'])

    def test_source_stays_fixed_and_invalid_point_rejected(self):
        run = engine.replay(self.data, self.settings)
        point = engine.point_evidence(run, self.meta, 0, len(run['claims']), candidate=1)
        self.assertEqual(point['visible'], 0)
        self.assertEqual(point['path'], [0])
        self.assertEqual(point['route']['path'], [])
        with self.assertRaises(ValueError):
            engine.point_evidence(run, self.meta, -1, 0)

    def test_recorded_trace_survives_an_experiment(self):
        recorded = engine.replay(self.data, self.settings)
        zero_settings = {**self.settings, 'reach': 0.}
        active = engine.replay({**self.data, 'xy': self.data['X'], 'pca': self.data['X']}, zero_settings)
        payload = engine.serialize_case(active, dict(settings=zero_settings), recorded)
        self.assertEqual(len(payload['claims']), 0)
        self.assertGreater(len(payload['recorded_claims']), 0)
        np.testing.assert_array_equal(payload['recorded_claims'], recorded['claims'])
        labels = self.data['base'].copy()
        for node, cluster, _, edge, parent in payload['recorded_claims']:
            self.assertIn(int(parent), self.data['edges'][int(edge)])
            labels[int(node)] = int(cluster)
        np.testing.assert_array_equal(labels, payload['recorded_final'])

    def test_multihop_path_has_exact_frame_visibility(self):
        run = engine.replay(self.data, self.settings)
        node = next(int(c[0]) for c in run['claims'] if len(engine.path_to(run, int(c[0]))) > 2)
        final = engine.point_evidence(run, self.meta, node, len(run['claims']))
        for count in range(len(run['claims']) + 1):
            point = engine.point_evidence(run, self.meta, node, count)
            visible_hops = [h for h in point['hops'] if h['claim'] <= count]
            self.assertEqual(point['visible_path'], [final['path'][0]] + [h['node'] for h in visible_hops])
            for hop in visible_hops:
                self.assertEqual(point['path'][point['path'].index(hop['node']) - 1], hop['parent'])


if __name__ == '__main__':
    unittest.main()
