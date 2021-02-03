from stannumpyro.dppl import NumPyroModel
from jax import random
# from cmdstanpy import CmdStanModel
from stanpyro_cuda.dppl import PyroModel



stanfile = "8schools.stan"
data = {
    "J": 8,
    "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
    "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
}

configs = {
    "warmups":15,
    "samples":30,
    "thin":3,
    "chains":2,
}


stanfile = "8schools.stan"


if __name__ == "__main__":

    # model = CmdStanModel(stan_file=stanfile)
    # fit = model.sample(
    #     data=data,
    #     iter_warmup=configs["warmups"],
    #     iter_sampling=configs["samples"],
    #     thin=configs["thin"],
    #     chains=configs["chains"],
    # )   


    # sshape = fit.stan_variable('mu').shape

    model = PyroModel(stanfile)
    mcmc = model.mcmc(**configs)
    mcmc.run(data)
    samples = mcmc.get_samples()
    pshape = mcmc.get_samples()["mu"].shape

    print(mcmc.summary())


    # model = NumPyroModel(stanfile)
    # mcmc = model.mcmc(**configs, progress_bar=False)
    # mcmc.run(random.PRNGKey(0), data)
    # npshape = mcmc.get_samples()["mu"].shape
    

    # print(f"Stan: {sshape}, Pyro: {pshape}, Numpyro: {npshape}")

