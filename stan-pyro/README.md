# Stan-Pyro

Python interface for the Stan to Pyro compiler.

## Install

```
pip install .
```

This will install two packages:
- `stanpyro` the pyro runtime and libraries on CPU.
- `stanpyro-cuda` the pyro runtime and libraries on GPU.

These two packages implement the same API.

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
from stanpyro import PyroModel

if __name__ == "__main__":

    stanfile = "8schools.stan"
    data = {
        'J': 8,
        'y': [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        'sigma': [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    }

    model = PyroModel(stanfile)
    mcmc = model.mcmc(
        samples = 1000,
        warmups = 100,
        chains=2,
        thin=2,
    )
    mcmc.run(data)
    print(mcmc.summary())
```

## API Reference

### PyroModel

```python
class PyroModel(stanfile, recompile=True, mode="comprehensive", compiler=["stanc"], build_dir="_tmp")
```

Simplified interface to compile and run Stan models using the [Stan to Pyro compiler](https://github.com/deepppl/stanc3)

**Parameters**
- `stanfile`: location of the stan file
- `recompile`: Whether to recompile the model (default to `True`)
- `mode`: compilation mode. One of `"comprehensive"`, `"mixed"`, or `"generative"` (default `"comprehensive"`)
- `compiler`: compiler command as a list (default `["stanc"]`)
- `build_dir`: name of the build directory (default `"_tmp"`)


```python
mcmc(samples, warmups=0, chains=1, thin=1, kernel=None, **kwargs) -> MCMCProxy
```

Provides access to Markov Chain Monte Carlo inference algorithms in NumPyro (see http://docs.pyro.ai/en/stable/mcmc.html).
The default kernel is NUTS.

**Parameters**
- `samples`: Number of samples to generate from the Markov chain
- `warmups`: Number of warmup steps (default `0`)
- `chains`: Number of MCMC chains to run (default `1`)
- `thin`: Positive integer that controls the fraction of post-warmup samples that are retained (default `1`)
- `**kwargs`: Other options that are directly passed to Pyro MCMC constructor

```python
svi(optim, loss) -> SVIProxy
```
Provide access to Stochastic Variational Inference given an ELBO loss objective (see http://docs.pyro.ai/en/stable/inference_algos.html)

**Parameters**
- `optim`:  a wrapper a for a PyTorch optimizer
- `loss`: an instance of a subclass of ELBO

### MCMCProxy

```python
class MCMCProxy(mcmc, module)
```

Wrapper for Pyro MCMC (see http://docs.pyro.ai/en/stable/mcmc.html).

**Parameters**
- `mcmc`: An instance of Numpyro MCMC (obtained from the `mcmc` method of `PyroModel`)
- `module`: The module containing the compiled code (see `compile` function)

```python
run(kwargs):
```

Run the inference

**Parameters**
- `kwargs`: Data passed as a dictionary.

```python
get_samples()
```

Get samples from the MCMC run.

```python
summary(prob=0.9)
```

Print the statistics of posterior samples collected during running this MCMC instance

**Parameters**
- `prob`: the probability mass of samples within the credible interval.

### SVIProxy

```python
class SVIProxy(svi, module)
```

Wrapper for NumPyro SVI (see http://num.pyro.ai/en/stable/svi.html).

**Parameters**
- `svi`: An instance of Numpyro SVI (obtained from the `svi` method of `NumPyroModel`)
- `module`: The module containing the compiled code (see `compile` function)

```python
preprocess(kwargs)
```

Preprocess the data to a format that can be used by Pyro.

**Parameters**
- `kwargs`: Data passed as a dictionary.

```python
step(kwargs)
```

Run one SVI step.

**Parameters**
- `kwargs`: Data passed as a dictionary (the results of `preprocess`).

```python
run(num_steps, kwargs):
```

Run multiple SVI steps.

**Parameters**
- `num_steps`: Number of steps
- `kwargs`: Data passed as a dictionary (the results of `preprocess`).

```python
sample_posterior(n, kwargs)
```

Generate samples from the posterior distribution

**Parameters**
- `n`: Number of samples
- `kwargs`: Data passed as a dictionary (the results of `preprocess`).


### Compile

```python
compile(mode, stanfile, compiler=["stanc"], recompile=True, build_dir="_tmp")
```

Compile a stan model to NumPyro.

**Parameters**
- `mode`: compilation mode. One of `"comprehensive"`, `"mixed"`, or `"generative"`
- `stanfile`: location of the stan file
- `compiler`: compiler command as a list (default `["stanc"]`)
- `recompile`: Whether to recompile the model (default to `True`)
- `build_dir`: name of the build directory (default `"_tmp"`)

This function generate a python file named after the Stan file in the build directory containing the generated code.
This file can then be used a python module.

### Distributions and StanLib

The files `distributions.py` and `stanlib.py` contains the NumPyro implementation of the Stan standard library.
The complete documentation can be found here: https://mc-stan.org/docs/2_26/functions-reference/index.html
