# Stan-NumPyro

Python interface for the Stan to NumPyro compiler.

## Install

```
pip install .
```

## Getting Started

Let start with the simple eight schools example from Gelman et al (Bayesian Data Analysis: Sec. 5.5, 2003). First save the following Stan code, e.g., in a file 8schools.stan:

```stan
data {
  int <lower=0> J; // number of schools
  real y[J]; // estimated treatment
  real<lower=0> sigma[J]; // std of estimated effect
}
parameters {
  real theta[J]; // treatment effect in school j
  real mu; // hyper-parameter of mean
  real<lower=0> tau; // hyper-parameter of sdv
}
model {
  tau ~ cauchy(0, 5); // a non-informative prior
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
  mu ~ normal(0, 5);
}
```

Then to compile and run inference with the NumPyro runtime:

```python
from stannumpyro.dppl import NumPyroModel
from jax import random

if __name__ == "__main__":

    stanfile = "8schools.stan"
    data = {
        'J': 8,
        'y': [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        'sigma': [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    }

    model = NumPyroModel(stanfile)
    mcmc = model.mcmc(
        samples = 1000,
        warmups = 100,
        chains=2,
        thin=2,
    )
    mcmc.run(random.PRNGKey(0), data)
    print(mcmc.summary())
```

The `NumPyroModel` constructor compiles and load the model.
The rest of the API is similar to [NumPyro](http://num.pyro.ai/en/stable/api.html).
Note that `mcmc.run` requires an explicit random seed as a first argument.

## API Reference

### NumPyroModel

```
class NumPyroModel(stanfile, recompile=True, mode="comprehensive", compiler=["stanc"], build_dir="_tmp")
```

Simplified interface to compile and run Stan models using the [Stan to NumPyro compiler](https://github.com/deepppl/stanc3)

**Parameters**
- `stanfile`: location of the stan file
- `recompile`: Whether to recompile the model (default to `True`)
- `mode`: compilation mode. One of `"comprehensive"`, `"mixed"`, or `"generative"` (default `"generative"`)
- `compiler`: compiler command as a list (default `["stanc"]`)
- `build_dir`: name of the build directory (default `"_tmp"`)


```
mcmc(samples, warmups=0, chains=1, thin=1, kernel=None, **kwargs) -> MCMCProxy
```

Provides access to Markov Chain Monte Carlo inference algorithms in NumPyro (see http://num.pyro.ai/en/stable/mcmc.html).
The default kernel is NUTS.

**Parameters**
- `samples`: Number of samples to generate from the Markov chain
- `warmups`: Number of warmup steps (default `0`)
- `chains`: Number of MCMC chains to run (default `1`)
- `thin`: Positive integer that controls the fraction of post-warmup samples that are retained (default `1`)
- `**kwargs`: Other options that are directly passed to NumPyro MCMC constructor 

```
svi(optim, loss) -> SVIProxy
```
Provide access to Stochastic Variational Inference given an ELBO loss objective (see http://num.pyro.ai/en/stable/svi.html)

**Parameters**
- `optim`: an instance of `_NumpyroOptim`
- `loss`: ELBO loss, i.e. negative Evidence Lower Bound, to minimize

### MCMCProxy

```
class MCMCProxy(mcmc, module)
```

Wrapper for NumPyro MCMC (see http://num.pyro.ai/en/stable/mcmc.html).

**Parameters**
- `mcmc`: An instance of NumPyro MCMC (obtained from the `mcmc` method of `NumPyroModel`)
- `module`: The module containing the compiled code (see `compile` function)

```
run(rng_key, kwargs):
```

Run the inference

**Parameters**
- `rng_key`: Random number generator key to be used for the sampling (e.g., `jax.random.PRNGKey(0)`).
- `kwargs`: Data passed as a dictionary.

```
get_samples()
```

Get samples from the MCMC run.

```
summary(prob=0.9)
```

Print the statistics of posterior samples collected during running this MCMC instance

**Parameters**
- `prob`: the probability mass of samples within the credible interval.

### SVIProxy

```
class SVIProxy(svi, module)
```

Wrapper for NumPyro SVI (see http://num.pyro.ai/en/stable/svi.html).
:warning: Still experimental.

**Parameters**
- `svi`: An instance of NumPyro SVI (obtained from the `svi` method of `NumPyroModel`)
- `module`: The module containing the compiled code (see `compile` function)

### Compile

```
compile(mode, stanfile, compiler=["stanc"], recompile=True, build_dir="_tmp")
```

Compile a stan model to NumPyro.

**Parameters**
- `mode`: compilation mode. One of `"comprehensive"`, `"mixed"`, or `"generative"` (default `"generative"`)
- `stanfile`: location of the stan file
- `compiler`: compiler command as a list (default `["stanc"]`)
- `recompile`: Whether to recompile the model (default to `True`)
- `build_dir`: name of the build directory (default `"_tmp"`)

This function generate a python file named after the stan file in the build directory containing the generated code.
This file can then be used a python module.

### Distributions and StanLib

The files `distributions.py` and `stanlib.py` contains the NumPyro implementation of the Stan standard library.
The complete documentation can be found here: https://mc-stan.org/docs/2_26/functions-reference/discrete-distributions.html
