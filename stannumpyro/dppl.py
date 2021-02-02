import os, sys, pathlib, importlib, subprocess
import numpyro, jax
import jax.numpy as jnp
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
                f"--numpyro",
                "--mode",
                mode,
                "--o",
                pyfile,
                stanfile,
            ]
        )
        # _exec(compiler + f" --numpyro --mode {mode} --o {pyfile} {stanfile}")
    return modname


class NumPyroModel:
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
            kernel = numpyro.infer.NUTS(
                self.module.model,
                adapt_step_size=True,
                init_strategy=numpyro.infer.initialization.init_to_sample,
            )

            mcmc = numpyro.infer.MCMC(
                kernel, warmups, samples, num_chains=chains, thinning=thin, **kwargs
            )
        return MCMCProxy(mcmc, self.module)

    # def svi(self, optimizer, loss=None):
    #     loss = loss if loss is not None else self.pyro.infer.Trace_ELBO()
    #     svi = self.pyro.infer.SVI(self.module.model, self.module.guide, optimizer, loss)
    #     if self.pyro_backend == "numpyro":
    #         svi.run = partial(svi.run, jax.random.PRNGKey(0))
    #     elif self.pyro_backend == "pyro":
    #         def run(self, num_step, *args, **kwargs):
    #             for _ in range(num_step):
    #                 svi.step(*args, **kwargs)
    #         svi.run = run
    #     return SVIProxy(svi, self.module)


class MCMCProxy:
    def __init__(self, mcmc, module):
        self.mcmc = mcmc
        self.module = module
        self.kwargs = {}

    def run(self, rng_key, kwargs):
        self.kwargs = self.module.convert_inputs(kwargs)
        if hasattr(self.module, "transformed_data"):
            self.kwargs.update(self.module.transformed_data(**self.kwargs))
        self.mcmc.run(rng_key, **self.kwargs)
        self.samples = self.mcmc.get_samples()
        if hasattr(self.module, "generated_quantities"):
            gen = self.module.map_generated_quantities(self.samples, **self.kwargs)
            self.samples.update(gen)

    def get_samples(self):
        return self.samples

    def summary(self):
        d_mean = _flatten_dict(
            {k: jnp.mean(jnp.array(v), axis=0) for k, v in self.samples.items()}
        )
        d_std = _flatten_dict(
            {k: jnp.std(jnp.array(v), axis=0) for k, v in self.samples.items()}
        )
        return DataFrame({"mean": Series(d_mean), "std": Series(d_std)})


# class SVIProxy(object):
#     def __init__(self, svi, module):
#         self.svi = svi
#         self.module = module
#         self.args = []
#         self.kwargs = {}

#     def posterior(self, n):
#         from numpyro import handlers
#         with handlers.seed(rng_seed=0):
#             return [self.svi.guide(**self.kwargs) for _ in range(n)]

#     def step(self, *args, **kwargs):
#         self.kwargs = kwargs
#         if hasattr(self.module, "transformed_data"):
#             self.kwargs.update(self.module.transformed_data(**self.kwargs))
#         return self.svi.step(**self.kwargs)

#     def run(self, num_steps, kwargs):
#         self.kwargs = self.module.convert_inputs(kwargs)
#         if hasattr(self.module, "transformed_data"):
#             self.kwargs.update(self.module.transformed_data(**self.kwargs))
#         return self.svi.run(num_steps, **self.kwargs)
