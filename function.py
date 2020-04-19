import torch


class MODEL:
    def __init__(self, density=None, log_density=None,
                 log_density_gradient=None):
        if (density is None) and (log_density is None):
            raise TypeError

        if (density is not None) and (log_density is not None):
            self.density_func = density
            self.log_density_func = log_density
        elif density is None:
            self.density_func = lambda x: torch.exp(log_density(x))
            self.log_density_func = log_density
        else:
            self.density_func = density
            self.log_density_func = lambda x: torch.log(density(x))

        if log_density_gradient is not None:
            self.log_density_gradient_func = log_density_gradient
        else:
            def log_density_gradient_func(x):
                x = torch.tensor(x.copy(), requires_grad=True)
                w = self.log_density_func(x)
                w.backward()
                return x.copy()
            self.log_density_gradient_func = log_density_gradient_func

    def log_gradient(self, x):
        return self.log_density_gradient_func(x)

    def density(self, x):
        return self.density_func(x)

    def log_density(self, x):
        return self.log_density_func(x)
