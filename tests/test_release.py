"""Regression checks for public configuration and XAI data plumbing; no GPU or API calls."""
import contextlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from HGS_XAI import load_xai_data, split_xai_data
from utils.config_loader import ConfigLoader

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ReleaseRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=['g1', 'g2', 'unmapped'])
        self.frame.index = [f'patient-{i}' for i in range(40)]
        self.frame['time'] = np.arange(1, 41)
        self.frame['event'] = np.tile([0, 1], 20)

    def test_private_config_override_resolves_data_paths(self):
        config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
        config['paths']['data_root'] = str(self.directory / 'inputs')
        local = self.directory / 'private.yaml'
        local.write_text(yaml.safe_dump(config))
        with patch.dict(os.environ, {'HGS_CONFIG': str(local)}):
            loader = ConfigLoader()
            self.assertEqual(loader.get_data_path('HCC'), str(self.directory / 'inputs/PRO/HCC/dataset.csv'))
            self.assertEqual(ConfigLoader(str(ROOT / 'config.yaml')).get_paths()['data_root'], 'data')

    def test_pro_clustering_does_not_require_reactome_layer(self):
        data_file = self.directory / 'dataset.csv'
        self.frame.to_csv(data_file)
        data, prior = load_xai_data('PRO', str(data_file), {'cox_num': 3}, {'type_know': 'hcluster'},
                                   {'reactome_graph_template': str(self.directory / 'absent{layer}')})
        pd.testing.assert_frame_equal(data, self.frame)
        self.assertIsNone(prior)

    def test_reactome_prior_survives_xai_split(self):
        prior = pd.DataFrame([[1, 0], [0, 1], [1, 1]], index=self.frame.columns[:-2])
        train, valid, test, _, result = split_xai_data(self.frame, 'TEST', {'type_know': 'Reactome'}, 7, prior)
        self.assertIs(result, prior)
        self.assertEqual((len(train), len(valid), len(test)), (24, 8, 8))
        self.assertEqual(train.shape[1] - 2, len(result))

    def test_missing_reactome_prior_fails_before_model_training(self):
        with self.assertRaisesRegex(ValueError, 'incidence matrix'):
            split_xai_data(self.frame, 'TEST', {'type_know': 'Reactome'}, 7)

    def test_string_split_keeps_node_and_feature_order_aligned(self):
        root = self.directory / 'data/PriorKnow/STRING'
        (root / 'sorted/layer').mkdir(parents=True)
        pd.DataFrame({'protein_id': ['g2', 'g1']}).to_csv(root / 'clusters.protein.ensg.csv', index=False)
        pd.DataFrame({'edge': [1, 1]}, index=['g2', 'g1']).to_csv(root / 'sorted/layer/TEST-Level1-H.csv')
        with working_directory(self.directory):
            train, valid, test, _, prior = split_xai_data(self.frame, 'TEST',
                {'type_know': 'STRING', 'method': 'layer', 'layer_STRING': 1}, 7)
        self.assertEqual(prior.index.tolist(), ['g1', 'g2'])
        self.assertEqual([a.shape[1] for a in (train, valid, test)], [4, 4, 4])
        # Time identifies the source row; check that both retained features still match it.
        for split in (train, valid, test):
            for row in split:
                np.testing.assert_allclose(row[:2], self.frame.iloc[int(row[-2]) - 1][['g1', 'g2']])

    def test_rna_reactome_loader_receives_directory_and_configured_survival(self):
        cohort = self.directory / 'RNA/TEST'
        prior_dir = self.directory / 'Reactome/reactome_P11'
        cohort.mkdir(parents=True)
        prior_dir.mkdir(parents=True)
        self.frame.iloc[:, :3].T.to_csv(cohort / 'feature_matrix.csv')
        pd.DataFrame(index=self.frame.columns[:3]).to_csv(cohort / 'genes_list.csv')
        pd.DataFrame({'edge': [1, 1, 1]}, index=self.frame.columns[:3]).to_csv(prior_dir / 'H1.csv')
        clinical = self.directory / 'survival.csv'
        pd.DataFrame({'PatientID': self.frame.index, 'OS Status': self.frame.event,
                      'OS_60': self.frame.time}).to_csv(clinical)
        with patch('utils.data_utils.config') as config:
            config.get_paths.return_value = {'survival_data_template': str(clinical)}
            data, prior = load_xai_data('RNA', str(cohort / 'feature_matrix.csv'),
                {'cox_num': 3, 'end_point': 'OS_60'}, {'type_know': 'Reactome', 'layer_Reactome': 11},
                {'reactome_graph_template': str(self.directory / 'Reactome/reactome_P{layer}')})
        self.assertEqual(data.shape, (40, 5))
        self.assertEqual(prior.shape, (3, 1))


if __name__ == '__main__':
    unittest.main()
