#--------------------- Stochastic Modeling Module-----------------------
'''
This module has classes and functions related to the stochastic modeling
of stochastic processes. In particular, the main classes are related to
the Geometric Brownian Motion Monte Carlo simulation.
'''

# ------------------------1. Libraries----------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from dateutil.relativedelta import relativedelta

#-------------------------2. Classes------------------------------------
class GBM(object):
    '''This objects has methods and attributes needed to simulate an 
    stochastic process as a Geometric Brownian Motion.
    '''
    def __init__(self, s, mu, sigma, Np=1000, T=5, Nt=60):
        """This functions initiates an instance of a GBM object.

        Inputs:
        -------
        s: numerical value
            Initial value of the Geometric Brownian Motions, from which
            the rest of the process will be simulated.
        mu: numerical value
            Mean or shift of the stochastic process.
        sigma: numerical value
            Standard deviation or volatility of the stochastic process.
        Np: int
            Number of paths that will be simulated.
        T: numerical value 
            Number of years or reference unit that will be simulated.
        Nt: int
            Number of steps taken during T.

        Outputs:
        --------
        None 
        """
        self.s = s
        self.mu = mu
        self.sigma = sigma
        self.Np = Np
        self.T = T
        self.Nt = Nt
        self.dt = T/Nt
        self.s_sims = self.simulate_paths()
    
    def simulate_paths(self):
        """Simulates Np paths of a Geometric Brownian Motion stochastic
        process. This simulation follows the logarithmic transformation
        on the GBM.
        """
        brownian_motion = np.random.normal(
            scale = self.dt**0.5,
            size = (self.Nt, self.Np)
        )
        first_term = (self.mu-self.sigma**2/2)*self.dt
        s_sims = self.s*np.exp(
            (first_term+self.sigma*brownian_motion).cumsum(axis=0))
        return s_sims
    
class GBMRateSeries(object):
    """This object stores a rate series, its statistical information
    and with it generates GBM simulations of the rate series. It also 
    has methods to plot historical behaviour plus its variations.
    """
    def __init__(self, series, Np=1000, Nt=60, T=5):
        """Initiates an instance of RateSeries object.
        
        Inputs:
        -------
        series: pandas Series instance.
            Series with the historical values of the interest variable.
            It is supposed to have a daily frequency.
        """
        # Define attributes of the object
        self.series = series

        # Log-monthly changes:
        log_changes = np.log(self.series/self.series.shift(1)).dropna()
        self.mu = log_changes.mean()
        self.sigma = log_changes.std()
        self.last = series.last('1D').item()
        self.Np = Np
        self.Nt = Nt
        self.T = T

        # Rate simulation:
        sim_df = self.simulate_rate()
        sim_df.loc[self.series.index[-1],:] = self.last
        self.sim_df = sim_df.sort_index()


    def simulate_rate(self):
        # Generate the GBM simulation for the rate
        sim_date_index = pd.date_range(
            start = self.series.index[-1]+relativedelta(months=1),
            end = self.series.index[-1]+relativedelta(months=self.Nt),
            freq='M'
        )
        rate_gbm = GBM(
            s = self.last,
            mu = self.mu,
            sigma = self.sigma,
            Np = self.Np,
            T = self.T,
            Nt = self.Nt
        )
        self.dt = rate_gbm.dt
        sim_paths = rate_gbm.s_sims
        sim_df = pd.DataFrame(data=sim_paths, index=sim_date_index)
        return sim_df
    
    def get_full_df(self):
        full_observed_df = pd.DataFrame(
            data = np.tile(self.series, (self.Np,1)).T,
            index = self.series.index
        )
        full_df = pd.concat([full_observed_df, self.sim_df])
        return full_df
    
    def get_monthly_variations(self):
        month_delta = self.series/self.series.shift(1)-1
        return month_delta
    
    def get_annual_variations(self):
        annual_delta = self.series/self.series.shift(12)-1
        return annual_delta

    def plot_full_series(self, start, end, historical, forecasts):
        date_max = historical.idxmax()
        max_value = historical.max()
        date_min = historical.idxmin()
        min_value = historical.min()
        #######self.series[start:].plot(color=)


#-----------------------3. Auxiliary Functions -------------------------
def quant_5(x):
    return x.quantile(0.05)

def quant_95(x):
    return x.quantile(0.95)



