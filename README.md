# Stan(Num)Pyro

Python interface for the Stan to (Num)Pyro compiler.

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

Then to compile and run inference:

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

    model = NumpyroModel(stanfile)
    mcmc = model.mcmc(
        samples = 1000,
        warmups = 100,
        chains=2,
        thin=2,
    )
    mcmc.run(random.PRNGKey(0), data)
    print(mcmc.summary())
```