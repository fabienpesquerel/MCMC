import torch

########################################
#                 ULA                  #
########################################


class ULA:
    def __init__(self, model, metric, gradient_step, random_step,
                 initialization):
        self.model = model
        self.metric = random_step * torch.sqrt(torch.tensor(2.)) * metric
        self.gamma = gradient_step
        self.current_point = initialization

    def proposition(self, x):
        prop = self.metric @ torch.randn_like(x)
        prop += x + self.gamma * self.model.log_gradient(x)
        return prop

    def unadjusted_step(self):
        proposition = self.proposition(self.current_point)
        self.current_point = proposition

    def fit(self, nbr_samples=500):
        generated_samples = 0
        samples = [self.current_point.numpy()]
        while generated_samples < nbr_samples:
            self.unadjusted_step()
            samples.append(self.current_point.numpy())
            generated_samples += 1
        return samples

########################################
#                MALA                  #
########################################


class MALA:
    def __init__(self, model, metric, gradient_step, random_step,
                 initialization):
        self.model = model
        self.metric = random_step * torch.sqrt(torch.tensor(2.)) * metric
        self.gamma = gradient_step
        self.current_point = initialization

    def proposition(self, x):
        prop = self.metric @ torch.randn_like(x)
        prop += x + self.gamma * self.model.log_gradient(x)
        return prop

    def adjusted_step(self):
        threshold = torch.rand(1).item()
        proposition = self.proposition(self.current_point)
        alpha = self.model.density(proposition) / \
            self.model.density(self.current_point)
        alpha = alpha.item()
        try:
            alpha = min(1., alpha)
        except:
            print("Numerical Error")
            alpha = 1.

        if threshold < alpha:
            self.current_point = proposition

        return (threshold < alpha)

    def fit(self, nbr_samples=1000):
        generated_samples = 0
        rejected_samples = 0
        samples = [self.current_point.numpy()]
        while generated_samples < nbr_samples:
            if self.adjusted_step():
                samples.append(self.current_point.numpy())
                generated_samples += 1
            else:
                rejected_samples += 1
        print(f"Acceptation ratio : \
{generated_samples/(generated_samples + rejected_samples)}")
        return samples

    def adjusted_fit(self, nbr_samples=1000, acceptation_ratio=0.63,
                     update=200, increase_factor=1.2, decrease_factor=0.9,
                     increase_acceleration=1.05, decrease_acceleration=0.95,
                     verbose=True):
        # Home made scheme - No theoretical guarantee that this works better
        # then the classical scheme (i.e. faster convergence).
        # Theoretical conditions for convergence are satisfied however
        generated_samples = 0
        rejected_samples = 0
        samples = [self.current_point.numpy()]
        while generated_samples < nbr_samples:
            if self.adjusted_step():
                samples.append(self.current_point.numpy())
                generated_samples += 1
            else:
                rejected_samples += 1
            if (generated_samples + rejected_samples) % update == 0:
                ratio = generated_samples / \
                    (generated_samples + rejected_samples)
                if ratio > acceptation_ratio:
                    # increase exploration
                    self.metric *= increase_factor
                    # next time, increase exploration even more
                    increase_factor *= increase_acceleration
                    decrease_factor *= increase_acceleration
                    # decrease_acceleration cannot be greater than 1
                    decrease_factor = min(decrease_factor, 1. - 1e-7)

                else:
                    # decrease exploration
                    self.metric *= decrease_factor
                    decrease_factor *= decrease_acceleration
                    increase_factor *= decrease_acceleration
                    # increase_acceleration cannot be smaller than 1
                    increase_factor = max(increase_factor, 1 + 1e-7)
        if verbose:
            print(f"Acceptation ratio : \
            {generated_samples/(generated_samples + rejected_samples)}")
        return samples


########################################
#                MALA                  #
########################################


class GMMALA:
    def __init__(self, model, metric, gradient_step, random_step,
                 initialization):
        self.model = model
        self.metric = lambda x: random_step * \
            torch.sqrt(torch.tensor(2.)) * metric(x)
        self.gamma = gradient_step
        self.current_point = initialization

    def proposition(self, x):
        prop = self.metric(x) @ torch.randn_like(x)
        prop += x + self.gamma * self.model.log_gradient(x)
        return prop

    def adjusted_step(self):
        threshold = torch.rand(1).item()
        proposition = self.proposition(self.current_point)
        alpha = self.model.density(proposition) / \
            self.model.density(self.current_point)
        alpha = alpha.item()
        try:
            alpha = min(1., alpha)
        except:
            print("Numerical Error")
            alpha = 1.

        if threshold < alpha:
            self.current_point = proposition

        return (threshold < alpha)

    def fit(self, nbr_samples=1000):
        generated_samples = 0
        rejected_samples = 0
        samples = [self.current_point.numpy()]
        while generated_samples < nbr_samples:
            if self.adjusted_step():
                samples.append(self.current_point.numpy())
                generated_samples += 1
            else:
                rejected_samples += 1
        print(f"Acceptation ratio : \
{generated_samples/(generated_samples + rejected_samples)}")
        return samples
