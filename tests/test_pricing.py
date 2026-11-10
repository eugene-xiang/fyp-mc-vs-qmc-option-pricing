"""Small checks of the pricing mathematics and repeatability."""
from dataclasses import asdict
import csv
import json
import math
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from src.models import AsianContract, asset_paths, discounted_payoffs, deterministic_price, european_call_price
from src.pricing import (normal_transform, sample_normals, price_option, summarise, importance_payoffs, tune_shift, METHODS)
from src.experiments import run_experiment, run_sweep, summarise_sweep, extend_benchmark
from src.plotting import plot_results, plot_sweep, export_sweep_tables


class PricingTests(unittest.TestCase):
    def test_sweep_contracts_seeds_and_saved_summaries(self):
        """Overlapping stages must reuse contracts and keep all random streams separate."""
        settings = {'contract': asdict(AsianContract()), 'methods': list(METHODS),
                    'budgets': [16, 64], 'replications': 3, 'seed': 1,
                    'benchmark_points': 256, 'benchmark_replications': 4,
                    'benchmark_se_fraction': .1,
                    'importance_sampling': {'shift_candidates': [0., 1., 2.], 'pilot_points': 128}}
        sweep = {'moneyness_strikes': [90., 120.], 'moneyness_dimension': 4,
                 'dimension_strikes': [90., 120.], 'dimensions': [2, 4], 'seed': 12345}
        
        with tempfile.TemporaryDirectory() as temp:
            folder, summary = run_sweep(settings, sweep, temp)
            repeated_folder, repeated = run_sweep(settings, sweep, temp)
            self.assertNotEqual(folder, repeated_folder)
            self.assertEqual(summary, summarise_sweep(folder))
            self.assertEqual(len(summary), 4 * 2 * 6)
            self.assertEqual({(r['strike'], r['monitoring_dates']) for r in summary},
                             {(90., 2), (90., 4), (120., 2), (120., 4)})
            
            for a, b in zip(summary, repeated):
                self.assertEqual({k: v for k, v in a.items() if 'elapsed' not in k},
                                 {k: v for k, v in b.items() if 'elapsed' not in k})
            
            all_seeds = []
            for child in folder.iterdir():
                if not child.is_dir():
                    continue
                saved = json.loads((child / 'settings.json').read_text())
                with (child / 'replicates.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                all_seeds.append(saved['pilot']['seed'])
                all_seeds.extend(int(r['seed']) for r in rows)
                reference = np.mean([float(r['estimate']) for r in rows if r['phase'] == 'benchmark'])
                for result in [r for r in summary if r['contract'] == child.name]:
                    estimates = [float(r['estimate']) for r in rows if r['phase'] == 'evaluation'
                                 and r['method'] == result['method']
                                 and int(r['payoff_evaluations']) == result['payoff_budget']]
                    self.assertEqual(len(estimates), 3)
                    self.assertAlmostEqual(result['replicate_rmse'],
                                           np.sqrt(np.mean((np.array(estimates) - reference)**2)))
                    self.assertEqual(result['benchmark_mean'], reference)
            self.assertEqual(len(all_seeds), len(set(all_seeds)))
            self.assertEqual(len(all_seeds), 4 * (1 + 4 + 2 * 6 * 3))
            
            for artifact in plot_sweep(folder) + export_sweep_tables(folder):
                self.assertGreater(artifact.stat().st_size, 0)
            target = folder / 'strike-120-dates-4'
            original_csv = (target / 'replicates.csv').read_text()
            
            with self.assertRaises(ValueError):
                extend_benchmark(target, 4, sweep['seed'])
            refined = extend_benchmark(target, 4, 900000)
            self.assertTrue((target / 'replicates.csv').read_text().startswith(original_csv))
            self.assertEqual(refined['benchmark']['replications'], 8)
            
            # Replaying recorded settings must reproduce both the original evaluation streams and the additional benchmark streams.
            recorded = json.loads((target / 'settings.json').read_text())['experiment']
            replay_folder, replay = run_experiment(recorded, temp)
            self.assertEqual(refined['benchmark'], replay['benchmark'])
            self.assertEqual([r['replicate_rmse'] for r in refined['comparisons']],
                             [r['replicate_rmse'] for r in replay['comparisons']])
            root_settings = json.loads((folder / 'settings.json').read_text())
            self.assertEqual(root_settings['sweep']['benchmark_extensions'][target.name],
                             {'replications': 4, 'seed': 900000})
            self.assertEqual(len(summarise_sweep(folder)), len(summary))
            collision = {**sweep, 'benchmark_extensions': {
                target.name: {'replications': 4, 'seed': sweep['seed']}}}
            with self.assertRaises(ValueError):
                run_sweep(settings, collision, temp)
        self.assertEqual(settings['seed'], 1)
        self.assertEqual(settings['contract']['strike'], 100.)

    def test_path_and_payoff_calculation(self):
        """Compare the recursion with the formula at absolute monitoring times."""
        c = AsianContract(monitoring_dates=3)
        z = np.array([[.1, -.7, 1.2], [-1., .4, .3]])
        expected = np.empty_like(z)
        for i in range(2):
            for j in range(3):
                t = (j + 1) * c.maturity / 3
                w = math.sqrt(c.maturity / 3) * math.fsum(z[i, :j+1])
                expected[i,j] = c.spot * math.exp((c.rate-c.volatility**2/2)*t+c.volatility*w)
        np.testing.assert_allclose(asset_paths(z,c), expected, rtol=1e-14)
        
        c = AsianContract(spot=500, strike=100, rate=.07, maturity=2, monitoring_dates=2)
        np.testing.assert_allclose(discounted_payoffs([[110,130],[90,100]],c), [20*math.exp(-.14),0])

    def test_zero_volatility(self):
        for rate in [-.03,0,.05]:
            for strike in [80.,100.,120.]:
                c = AsianContract(volatility=0, rate=rate, strike=strike, monitoring_dates=16)
                for method in METHODS:
                    self.assertAlmostEqual(price_option(c,method,128,77)['estimate'], deterministic_price(c), places=12)

    def test_european_call(self):
        """With one monitoring date, simulated prices should agree with Black--Scholes."""
        c = AsianContract(monitoring_dates=1)
        reference = european_call_price(c)
        self.assertAlmostEqual(reference,10.450583572185565,places=11)
        for method in METHODS:
            n = 8192 if method.endswith('_av') else 16384
            shift = 1.0 if method.endswith('_is') else 0.0
            values = [price_option(c,method,n,729+i,shift=shift)['estimate'] for i in range(16)]
            result = summarise(values)
            self.assertLessEqual(abs(result['mean']-reference),6*result['mean_standard_error'])

    def test_normal_transform_and_complete_net(self):
        z,count = normal_transform([0,.25,.5,.75,1])
        self.assertEqual(count,2)
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_allclose(z,-z[::-1],atol=1e-14)
        z,count = sample_normals(256,4,'rqmc',782)
        self.assertEqual(count,0)
        
        for column in ndtr(z).T:
            np.testing.assert_array_equal(np.sort((column*256).astype(int)),np.arange(256))
        with self.assertRaises(ValueError): sample_normals(1000,4,'rqmc',782)
        with self.assertRaises(ValueError): normal_transform([float('nan')])
        with self.assertRaises(ValueError): AsianContract(volatility=-.1)
        c=AsianContract(monitoring_dates=8)
        a=price_option(c,'rqmc',4096,725,2**-53)
        b=price_option(c,'rqmc',4096,725,2**-40)
        self.assertEqual(a['estimate'],b['estimate'])
        self.assertEqual(a['endpoint_adjustments'],b['endpoint_adjustments'])

    def test_repeatability_and_uncertainty(self):
        c=AsianContract(monitoring_dates=8)
        
        for method in METHODS:
            a=price_option(c,method,1024,888)
            b=price_option(c,method,1024,888)
            self.assertEqual(a['estimate'],b['estimate'])
            self.assertEqual(a['iid_standard_error'] is None,method.startswith('rqmc'))
        result=summarise([1.,2.,3.],2.)
        self.assertEqual(result['estimator_variance'],1.)
        self.assertAlmostEqual(result['mean_standard_error'],1/math.sqrt(3))
        self.assertAlmostEqual(result['replicate_rmse'],math.sqrt(2/3))

    def test_antithetic_pairs_and_mc_uncertainty(self):
        """Independently calculate both paths at absolute times, then pair them."""
        c = AsianContract(monitoring_dates=3, rate=.03, volatility=.3)
        z = np.array([[.1, -.7, 1.2], [-1., .4, .3], [.8, 1.1, -.2], [-.6, -.2, -.5]])
        pairs = []
        for row in z:
            payoffs = []
            for sign in [1, -1]:
                prices = [c.spot * math.exp(
                    (c.rate - c.volatility**2 / 2) * (j + 1) * c.maturity / 3
                    + c.volatility * math.sqrt(c.maturity / 3) * sign * math.fsum(row[:j+1])
                ) for j in range(3)]
                payoffs.append(math.exp(-c.rate * c.maturity) * max(math.fsum(prices) / 3 - c.strike, 0))
            pairs.append(math.fsum(payoffs) / 2)
        expected_mean = math.fsum(pairs) / 4
        expected_se = math.sqrt(math.fsum((p - expected_mean)**2 for p in pairs) / 3 / 4)
        for method in ['mc_av', 'rqmc_av']:
            with patch('src.pricing.sample_normals', return_value=(z.copy(), 2)) as sampler:
                result = price_option(c, method, 4, 77)
            self.assertEqual(sampler.call_count, 1)
            self.assertEqual(sampler.call_args.args[:4], (4, 3, method.removesuffix('_av'), 77))
            self.assertAlmostEqual(result['estimate'], expected_mean, places=12)
            self.assertEqual(result['endpoint_adjustments'], 2)
            if method == 'mc_av':
                self.assertAlmostEqual(result['iid_standard_error'], expected_se, places=12)
            else:
                self.assertIsNone(result['iid_standard_error'])
        self.assertIsNone(price_option(c, 'mc_av', 1, 77)['iid_standard_error'])
        with self.assertRaises(ValueError):
            price_option(c, 'rqmc_av', 1000, 77)

    def test_antithetic_experiment_budgets_and_variance_ratios(self):
        settings = {'contract': asdict(AsianContract(monitoring_dates=4)),
                    'methods': ['mc', 'rqmc', 'mc_av', 'rqmc_av'], 'budgets': [16, 64], 'replications': 4,
                    'seed': 812, 'benchmark_points': 256, 'benchmark_replications': 4,
                    'benchmark_se_fraction': .1}
        with tempfile.TemporaryDirectory() as temp:
            first, summary = run_experiment(settings, temp)
            second, repeated = run_experiment(settings, temp)
            self.assertNotEqual(first, second)
            with (first / 'replicates.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            with (second / 'replicates.csv').open(newline='') as handle:
                repeated_rows = list(csv.DictReader(handle))
            for a, b in zip(rows, repeated_rows):
                a.pop('elapsed_seconds')
                b.pop('elapsed_seconds')
                self.assertEqual(a, b)
            self.assertEqual(len(rows), 4 + 2 * 4 * 4)
            self.assertEqual(len({row['seed'] for row in rows}), len(rows))
            for row in rows:
                n = int(row['base_points'])
                av = row['method'].endswith('_av')
                self.assertEqual(int(row['antithetic_pairs']), n if av else 0)
                self.assertEqual(int(row['payoff_evaluations']), 2 * n if av else n)
                if row['phase'] == 'evaluation':
                    self.assertIn(int(row['payoff_evaluations']), settings['budgets'])
                self.assertEqual(n & (n - 1), 0)
                if row['method'].startswith('rqmc'):
                    self.assertEqual(row['iid_standard_error'], '')
            self.assertEqual(len(summary['comparisons']), 8)
            for row in summary['comparisons']:
                baseline = next(b for b in summary['comparisons']
                                if b['method'] == row['method'].removesuffix('_av')
                                and b['payoff_budget'] == row['payoff_budget'])
                self.assertAlmostEqual(row['variance_reduction_factor'],
                                       baseline['estimator_variance'] / row['estimator_variance'])
            self.assertGreater(plot_results(first).stat().st_size, 0)
            
            # A zero payoff variance has no defined variance-reduction ratio
            settings['contract']['volatility'] = 0.0
            settings['contract']['strike'] = 1000.0
            _, zero_summary = run_experiment(settings, temp)
            for row in zero_summary['comparisons']:
                self.assertIsNone(row['variance_reduction_factor'])
                self.assertFalse(row['variance_reduction_factor_defined'])
            for bad_budget in [1, 3]:
                with self.assertRaises(ValueError):
                    run_experiment({**settings, 'budgets': [bad_budget]}, temp)

    def test_saved_experiment_repeats(self):
        settings={'contract':asdict(AsianContract(monitoring_dates=2)), 'budgets':[16,64],
                  'replications':3,'seed':454,'benchmark_points':256,'benchmark_replications':4,
                  'benchmark_se_fraction':.1}
        with tempfile.TemporaryDirectory() as temp:
            first,_=run_experiment(settings,temp)
            second,_=run_experiment(settings,temp)
            def read(folder):
                with (folder/'replicates.csv').open(newline='') as f:
                    return [{k:v for k,v in r.items() if k!='elapsed_seconds'} for r in csv.DictReader(f)]
            self.assertNotEqual(first,second)
            self.assertEqual(read(first),read(second))
            seeds=[r['seed'] for r in read(first)]
            self.assertEqual(len(seeds),len(set(seeds)))
            
            # The graph must be reproducible using CSV summaries alone
            with (first / 'summary.csv').open(newline='') as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 4)
            self.assertGreater(float(summary_rows[0]['benchmark_mean']), 0)
            figure = plot_results(first)
            self.assertGreater(figure.stat().st_size, 0)

    def test_importance_weights_against_density_ratio(self):
        """Compute p(X)/q(X) from normal densities, and paths at absolute times."""
        c = AsianContract(monitoring_dates=3)
        z = np.array([[.1, -.7, 1.2], [-1., .4, .3], [.8, 1.1, -.2], [-.6, -.2, -.5]])
        shift = 1.5
        theta = shift / math.sqrt(3)
        expected = []
        for row in z:
            x = row + theta
            p = math.prod(math.exp(-v*v/2) / math.sqrt(2*math.pi) for v in x)
            q = math.prod(math.exp(-(v-theta)**2/2) / math.sqrt(2*math.pi) for v in x)
            prices = [c.spot * math.exp( (c.rate - c.volatility**2/2) * (j+1)/3 + c.volatility / math.sqrt(3) * math.fsum(x[:j+1]) ) for j in range(3)]
            payoff = math.exp(-c.rate) * max(math.fsum(prices)/3 - c.strike, 0)
            expected.append(payoff * p / q)
        np.testing.assert_allclose(importance_payoffs(z, c, shift), expected, rtol=1e-13)
        for method in ['mc_is', 'rqmc_is']:
            with patch('src.pricing.sample_normals', return_value=(z.copy(), 0)):
                result = price_option(c, method, 4, 77, shift=shift)
            self.assertAlmostEqual(result['estimate'], np.mean(expected), places=12)
            if method == 'mc_is':
                self.assertAlmostEqual(result['iid_standard_error'], np.std(expected, ddof=1)/2, places=12)
            else:
                self.assertIsNone(result['iid_standard_error'])

    def test_importance_zero_shift_and_numerical_failures(self):
        c = AsianContract(monitoring_dates=4)
        for baseline in ['mc', 'rqmc']:
            a = price_option(c, baseline, 256, 991)
            b = price_option(c, baseline + '_is', 256, 991, shift=0)
            for key in ['estimate', 'iid_standard_error', 'endpoint_adjustments']:
                self.assertEqual(a[key], b[key])
        for shift in [-1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                importance_payoffs(np.zeros((4, 4)), c, shift)
        with self.assertRaises(ValueError):
            price_option(c, 'mc', 256, 991, shift=1)
        with self.assertRaises(ValueError):
            price_option(c, 'rqmc_is', 1000, 991, shift=1)
        
        # Underflow must not silently turn an unsuitable shift into zero variance
        with self.assertRaises(FloatingPointError):
            importance_payoffs(np.zeros((4, 4)), c, 100)

    def test_importance_zero_volatility_with_nonzero_shift(self):
        """A constant payoff times the density ratio integrates to that payoff."""
        c = AsianContract(volatility=0, monitoring_dates=1)
        for shift in [0.5, 1.5]:
            def integrand(z):
                weighted = importance_payoffs([[z]], c, shift)[0]
                return weighted * math.exp(-z*z/2) / math.sqrt(2*math.pi)
            
            # The omitted normal tails are negligible for these finite shifts
            actual, _ = quad(integrand, -12, 12, epsabs=1e-10)
            self.assertAlmostEqual(actual, deterministic_price(c), places=9)

    def test_pilot_selection_and_uninformative_candidates(self):
        c = AsianContract(monitoring_dates=4)
        pilot = tune_shift(c, [0., .5, 1., 1.5, 2.], 256, 878)
        repeated = tune_shift(c, [0., .5, 1., 1.5, 2.], 256, 878)
        for key in pilot:
            if key != 'elapsed_seconds':
                self.assertEqual(pilot[key], repeated[key])
        eligible = [r for r in pilot['candidates'] if r['positive_payoffs'] > 0]
        expected = min(eligible, key=lambda r: (r['second_moment'], r['shift']))
        self.assertEqual(pilot['selected_shift'], expected['shift'])
        self.assertEqual(pilot['payoff_evaluations'], 256 * 5)
        
        # An all-zero candidate must not beat a candidate with positive payoffs.
        rare = AsianContract(strike=150, monitoring_dates=4)
        with patch('src.pricing.sample_normals', return_value=(np.zeros((8, 4)), 0)):
            informative = tune_shift(rare, [0., 2., 4.], 8, 878)
        self.assertEqual(informative['selected_shift'], 4.)
        self.assertEqual(informative['candidates'][0]['positive_payoffs'], 0)
        zero = tune_shift(AsianContract(volatility=0, strike=1000), [0., 1.], 16, 879)
        self.assertEqual(zero['selected_shift'], 0.)
        self.assertFalse(zero['informative'])
        for candidates in [[1.], [0., 0.], [0., -1.], [0., float('inf')]]:
            with self.assertRaises(ValueError):
                tune_shift(c, candidates, 16, 878)

    def test_importance_experiment_pilot_separation_and_costs(self):
        settings = {'contract': asdict(AsianContract(monitoring_dates=4)),
                    'methods': list(METHODS), 'budgets': [16, 64], 'replications': 3,
                    'seed': 990, 'benchmark_points': 256, 'benchmark_replications': 4,
                    'benchmark_se_fraction': .1,
                    'importance_sampling': {'shift_candidates': [0., .5, 1.], 'pilot_points': 128}}
        with tempfile.TemporaryDirectory() as temp:
            first, summary = run_experiment(settings, temp)
            second, _ = run_experiment(settings, temp)
            saved = json.loads((first / 'settings.json').read_text())
            pilot = saved['pilot']
            with (first / 'replicates.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            with (second / 'replicates.csv').open(newline='') as f:
                repeated = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4 + 2 * 6 * 3)
            self.assertEqual(len({r['seed'] for r in rows}), len(rows))
            self.assertNotIn(str(pilot['seed']), {r['seed'] for r in rows})
            for a, b in zip(rows, repeated):
                a.pop('elapsed_seconds')
                b.pop('elapsed_seconds')
                self.assertEqual(a, b)
                is_method = a['method'].endswith('_is')
                self.assertEqual(float(a['shift']), pilot['selected_shift'] if is_method else 0)
                if is_method:
                    self.assertEqual(a['base_points'], a['payoff_evaluations'])
                    self.assertEqual(a['antithetic_pairs'], '0')
            for r in summary['comparisons']:
                is_method = r['method'].endswith('_is')
                pilot_count = 128 * 3 if is_method else 0
                pilot_time = pilot['elapsed_seconds'] if is_method else 0
                self.assertEqual(r['standalone_payoff_evaluations'], r['payoff_budget'] + pilot_count)
                self.assertAlmostEqual(r['standalone_elapsed_seconds'], r['mean_elapsed_seconds'] + pilot_time)
            self.assertGreater(plot_results(first).stat().st_size, 0)


if __name__=='__main__':
    unittest.main()
