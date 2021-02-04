import os, sys, pathlib, importlib, subprocess
import pyro, torch
from os.path import splitext, basename, dirname
from pandas import DataFrame, Series


def _flatten_dict(d):
    def _flatten(name, a):
        if len(a.shape) == 0:
            return {name: a.tolist()}
        else:
            return {
                k: v
                for d in (_flatten(name + f"[{i+1}]", v) for i, v in enumerate(a))
                for k, v in d.items()
            }

    return {
        fk: fv for f in (_flatten(k, v) for k, v in d.items()) for fk, fv in f.items()
    }


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
        self, stanfile, recompile=True, mode="comprehensive", compiler=["stanc"]
    ):

        if not os.path.exists("_tmp"):
            os.makedirs("_tmp")
            pathlib.Path("_tmp/__init__.py").touch()

        modname = compile(mode, stanfile, compiler, recompile, build_dir="_tmp")

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

    def summary(self):
        d_mean = _flatten_dict(
            {k: torch.mean(torch.tensor(v), axis=0) for k, v in self.samples.items()}
        )
        d_std = _flatten_dict(
            {k: torch.std(torch.tensor(v), axis=0) for k, v in self.samples.items()}
        )
        return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})


class SVIProxy(object):
    def __init__(self, svi, module):
        self.svi = svi
        self.module = module

    def preprocess(self, kwargs):
        kwargs.update(self.module.convert_inputs(kwargs))
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
