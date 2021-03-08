import os, sys, pathlib, importlib, subprocess
import pyro, torch
from os.path import splitext, basename, dirname
from pandas import DataFrame, Series
from itertools import product


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
                f"--pyro",
                "--mode",
                mode,
                "--o",
                pyfile,
                stanfile,
            ]
        )
    return modname


class PyroModel:
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
            kernel = pyro.infer.NUTS(
                self.module.model,
                adapt_step_size=True,
                init_strategy=pyro.infer.autoguide.initialization.init_to_sample,
            )

            mcmc = pyro.infer.MCMC(
                kernel, samples, warmup_steps=warmups, num_chains=chains, **kwargs
            )
        return MCMCProxy(mcmc, self.module, thin)

    def svi(self, optim, loss):
        svi = pyro.infer.SVI(self.module.model, self.module.guide, optim, loss)
        return SVIProxy(svi, self.module)


class MCMCProxy:
    def __init__(self, mcmc, module, thin):
        self.mcmc = mcmc
        self.module = module
        self.thin = thin

    def run(self, kwargs):
        kwargs = self.module.convert_inputs(kwargs)
        if hasattr(self.module, "transformed_data"):
            kwargs.update(self.module.transformed_data(**kwargs))
        self.mcmc.run(**kwargs)
        self.samples = self.mcmc.get_samples()
        if self.thin > 1:
            self.samples = {x: self.samples[x][:: self.thin] for x in self.samples}
        if hasattr(self.module, "generated_quantities"):
            gen = self.module.map_generated_quantities(self.samples, **kwargs)
            self.samples.update(gen)

    def get_samples(self):
        return self.samples

    def summary(self, prob=0.9):
        samples = {k:v.double() for k, v in self.samples.items()}
        summary_dict = pyro.infer.mcmc.util.summary(
            samples, prob=prob, group_by_chain=False
        )
        columns = list(summary_dict.values())[0].keys()
        index = []
        rows = []
        for name, stats_dict in summary_dict.items():
            shape = stats_dict["mean"].shape
            if len(shape) == 0:
                index.append(name)
                rows.append([v.numpy() for v in stats_dict.values()])
            else:
                for idx in product(*map(range, shape)):
                    idx_str = "[{}]".format(
                        ",".join(map(str, map(lambda i: i + 1, idx)))
                    )
                    index.append(name + idx_str)
                    rows.append([v[idx].numpy() for v in stats_dict.values()])
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

    def sample_posterior(self, n, kwargs):
        kwargs = self.preprocess(kwargs)
        return [self.svi.guide(**kwargs) for _ in range(n)]

    def step(self, kwargs):
        return self.svi.step(**kwargs)

    def run(self, num_steps, kwargs):
        kwargs = self.preprocess(kwargs)
        return self.svi.run(num_steps, **kwargs)
