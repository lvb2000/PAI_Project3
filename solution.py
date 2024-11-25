"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from matplotlib import pyplot as plt
import os


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        self.noise_f = 0.15
        self.noise_v = 0.0001
        self.domain = DOMAIN
        self.kappa = SAFETY_THRESHOLD
        self.lagrangian = 1

        # Data storage
        self.X = np.empty((0, 1))  # Empty array for the input data (X)
        self.y_f = np.empty((0, 1))  # Empty array for the objective function values
        self.y_v = np.empty((0, 1)) 

        self.kernel_f = Matern(nu=2.5) # RBF(length_scale=1.0)
        self.kernel_v = ConstantKernel(4.0, (1e-1,1e2)) + Matern(nu=2.5 ,length_scale=1, length_scale_bounds=(1e-3, 1e3))

        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f, alpha=self.noise_f**2, normalize_y=True, n_restarts_optimizer=10)
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v, alpha=self.noise_v**2, normalize_y=True, n_restarts_optimizer=10)

        # open a new directory for the plots at local machine
        if not os.path.exists("plots"):
            os.makedirs("plots")
        # Start with the base directory
        self.directory = "plots_1"
        counter = 1
        # Check if directory exists and increment the name if necessary
        while os.path.exists(f"plots/{self.directory}"):
            counter += 1
            self.directory = f"plots_{counter}"

        # Create the directory
        os.makedirs("plots/"+self.directory)
        print(f"Directory created: {self.directory}")
        self.plot_counter = 1


    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """

        return float(self.optimize_acquisition_function())

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        # Predict mean and standard deviation for logP (f)
        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        # This gives the point of maximum uncertainty
        sigma_f = np.maximum(sigma_f, 1e-9)  # Avoid division by zero

        # Predict mean and standard deviation for SA (v)
        mu_v, sigma_v = self.gp_v.predict(x, return_std=True)

        # Compute the worst case probability of feasibility: P(v(x) < kappa)
        relaxation = self.lagrangian * max(0, mu_v)

        # Compute Expected Improvement for f(x)
        y_max = np.max(self.y_f)
        z = (mu_f - y_max) / sigma_f
        ei = (mu_f - y_max) * norm.cdf(z) + sigma_f * norm.pdf(z)

        # Weight EI by the feasibility probability
        af_value = ei.flatten() - relaxation

        return af_value

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        x = np.array([[x]]).reshape(1, 1)  # Shape (1, 1)
        f = np.array([[f]]).reshape(1, 1)  # Shape (1, 1)
        v = np.array([[v]]).reshape(1, 1)  # Shape (1, 1)

        

        # Add new observation to the dataset
        self.X = np.vstack((self.X, x))
        self.y_f = np.vstack((self.y_f, f))
        self.y_v = np.vstack((self.y_v, v))

        # Refit GPs
        self.gp_f.fit(self.X, self.y_f)
        self.gp_v.fit(self.X, self.y_v)

        self.plot()



    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        X_samples = np.linspace(0, 10, 1000).reshape(-1, 1)
        mu_f, std = self.gp_f.predict(X_samples, return_std=True)
        mu_v, std = self.gp_v.predict(X_samples, return_std=True)

        # Filter out points where v(x) >= kappa
        feasible_indices = np.where(mu_v.flatten() < self.kappa)[0]

        if len(feasible_indices) == 0:
            raise ValueError("No feasible solution found within the constraint!")

        # Find the point with the maximum predicted f(x) among feasible points
        feasible_mu_f = mu_f[feasible_indices]
        feasible_X = X_samples[feasible_indices]

        x_opt = feasible_X[np.argmax(feasible_mu_f)].item()
        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        X_samples = np.linspace(0, 11, 1000).reshape(-1, 1)
        mu_f, std_f = self.gp_f.predict(X_samples, return_std=True)
        mu_v, std_v = self.gp_v.predict(X_samples, return_std=True)

        # Create figure and axis objects
        fig, ax1 = plt.subplots()

        X_samples = X_samples.flatten()

        # First curve on the left axis
        ax1.plot(X_samples, mu_f, label='Bioavailability Curve', color='blue')
        ax1.fill_between(X_samples, mu_f - std_f, mu_f + std_f, color='blue', alpha=0.2)
        ax1.set_xlabel('X-axis')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Second curve on the right axis
        ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis
        ax2.plot(X_samples, mu_v, label='Synthetic Accessibility Curve', color='red')
        ax2.fill_between(X_samples, mu_v - std_v, mu_v + std_v, color='red', alpha=0.2)
        ax2.tick_params(axis='y', labelcolor='red')

        # fix axis intervals
        #ax1.set_xlim(0, 11)
        #ax1.set_ylim(-5, 1)
        #ax2.set_ylim( 1, 4)

        # get the last new point
        new_point = self.X[-1]
        ax1.scatter(new_point, self.y_f[-1], color='green', label='New Point')

        # all points except the last one
        ax1.scatter(self.X[:-1], self.y_f[:-1], color='black', label='Old Points')

        # Title and layout adjustments
        plt.title('Posterior Predictions')
        fig.tight_layout()
        # plot
        #plt.show()
        # wait
        #plt.pause(0.3)
        # Save plot with increasing counter
        plt.savefig(f"plots/{self.directory}/plot_{self.plot_counter}.png")
        self.plot_counter += 1
        plt.close()


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    # add infeasible region
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)  # bioavailability f maps feature to logP (proxy of bioavailability)
    cost_val = v(x_init)  # synthetic accessibility (SA)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.rand()
        cost_val = v(x) + np.random.rand()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
