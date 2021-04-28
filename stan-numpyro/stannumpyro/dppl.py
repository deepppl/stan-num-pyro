import os, sys, pathlib, importlib, subprocess
import numpyro, jax
import jax.numpy as jnp
from os.path import splitext, basename, dirname
from pandas import DataFrame, Series
from itertools import product
from numpyro.handlers import seed, trace


def _exec(cmd):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"{stderr.decode('utf-8').strip()}")
    if stdout:
        print(f"stdout.decode('utf-8').strip()")
    return None


def compile(mode, stanfile, compiler=["stanc"], recompile=True, build_dir="_tmp"):
    name = splitext(basename(stanfile))[0]
    pyfile = f"{build_dir}/{name}.py"
    modname = f"{build_dir}.{name}"
    if recompile:
        _exec(
            compiler
            + [
                f"--numpyro",
                "--mode",
                mode,
                "--o",
                pyfile,
                stanfile,
            ]
        )
    return modname


class NumPyroModel:
    def __init__(
        self,
        stanfile,
        recompile=True,
        mode="comprehensive",
        compiler=["stanc"],
        build_dir="_tmp",
    ):

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
            if not os.path.exists(f"{build_dir}/__init__.py"):
                pathlib.Path(f"{build_dir}/__init__.py").touch()

        modname = compile(mode, stanfile, compiler, recompile, build_dir)

        self.module = importlib.import_module(modname)
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])

    def mcmc(self, samples, warmups=0, chains=1, thin=1, kernel=None, **kwargs):
        if kernel is None:
            kernel = numpyro.infer.NUTS(
                self.module.model,
                adapt_step_size=True,
                init_strategy=numpyro.infer.initialization.init_to_sample,
            )

        mcmc = numpyro.infer.MCMC(
            kernel, warmups, samples, num_chains=chains, thinning=thin, **kwargs
        )
        return MCMCProxy(mcmc, self.module)

    def svi(self, optim, loss, guide=None):
        if not guide:
            guide = self.module.guide
        svi = numpyro.infer.SVI(self.module.model, guide, optim, loss)
        return SVIProxy(svi, self.module)


class MCMCProxy:
    def __init__(self, mcmc, module):
        self.mcmc = mcmc
        self.module = module
        self.kwargs = {}

    def run(self, rng_key, kwargs):
        kwargs = self.module.convert_inputs(kwargs)
        if hasattr(self.module, "transformed_data"):
            kwargs.update(self.module.transformed_data(**kwargs))
        self.mcmc.run(rng_key, **kwargs)
        self.samples = self.mcmc.get_samples()
        if hasattr(self.module, "generated_quantities"):
            with numpyro.handlers.seed(rng_seed=0):
                gen = self.module.map_generated_quantities(self.samples, **kwargs)
                self.samples.update(gen)

    def get_samples(self):
        return self.samples

    def summary(self, prob=0.9):
        summary_dict = numpyro.diagnostics.summary(
            self.samples, prob=prob, group_by_chain=False
        )
        columns = list(summary_dict.values())[0].keys()
        index = []
        rows = []
        for name, stats_dict in summary_dict.items():
            shape = stats_dict["mean"].shape
            if len(shape) == 0:
                index.append(name)
                rows.append(stats_dict.values())
            else:
                for idx in product(*map(range, shape)):
                    idx_str = "[{}]".format(
                        ",".join(map(str, map(lambda i: i + 1, idx)))
                    )
                    index.append(name + idx_str)
                    rows.append([v[idx] for v in stats_dict.values()])
        return DataFrame(rows, columns=columns, index=index)


class SVIProxy(object):
    def __init__(self, svi, module):
        self.svi = svi
        self.module = module

    def preprocess(self, kwargs):
        kwargs = self.module.convert_inputs(kwargs)
        if hasattr(self.module, "transformed_data"):
            kwargs.update(self.module.transformed_data(**kwargs))
        return kwargs

    def get_samples(self, rng_key, params, num_samples):
        if hasattr(self.svi.guide, "sample_posterior"):
            return self.svi.guide.sample_posterior(
                rng_key, params, sample_shape=(num_samples,)
            )
        else:
            exec_trace = trace(seed(self.svi.model, rng_key)).get_trace()
            predictive = numpyro.infer.Predictive(
                self.svi.guide,
                params=params,
                num_samples=num_samples,
                return_sites=list(exec_trace.keys()),
            )
            return predictive(rng_key)

    def init(self, rng_key, kwargs):
        return self.svi.init(rng_key, **kwargs)

    def get_params(self, svi_state):
        return self.svi.get_params(svi_state)

    def update(self, svi_state, kwargs):
        return self.svi.update(svi_state, **kwargs)

    def stable_update(self, svi_state, kwargs):
        return self.svi.stable_update(svi_state, **kwargs)

    def run(self, rng_key, num_steps, num_samples, kwargs):
        kwargs = self.preprocess(kwargs)
        self.svi_results = self.svi.run(rng_key, num_steps, **kwargs)
        self.samples = self.get_samples(
            rng_key, self.svi_results.params, num_samples=num_samples
        )
        if hasattr(self.module, "generated_quantities"):
            with numpyro.handlers.seed(rng_seed=0):
                gen = self.module.map_generated_quantities(self.samples, **kwargs)
                self.samples.update(gen)

    def evaluate(self, svi_state, kwargs):
        return self.svi.evaluate(svi_state, **kwargs)

    def summary(self, samples=None, prob=0.9):
        if not samples:
            samples = self.samples
        summary_dict = numpyro.diagnostics.summary(
            samples, prob=prob, group_by_chain=False
        )
        columns = list(summary_dict.values())[0].keys()
        index = []
        rows = []
        for name, stats_dict in summary_dict.items():
            shape = stats_dict["mean"].shape
            if len(shape) == 0:
                index.append(name)
                rows.append(stats_dict.values())
            else:
                for idx in product(*map(range, shape)):
                    idx_str = "[{}]".format(
                        ",".join(map(str, map(lambda i: i + 1, idx)))
                    )
                    index.append(name + idx_str)
                    rows.append([v[idx] for v in stats_dict.values()])
        return DataFrame(rows, columns=columns, index=index)
