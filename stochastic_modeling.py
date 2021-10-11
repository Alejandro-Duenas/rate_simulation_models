#--------------------- Stochastic Modeling Module-----------------------
'''
This module has classes and functions related to the stochastic modeling
of stochastic processes. In particular, the main classes are related to
the Geometric Brownian Motion Monte Carlo simulation.
'''

# ------------------------1. Libraries----------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
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
    def __init__(self, series, Np=1000, Nt=60, T=5, color_dict=None):
        """Initiates an instance of RateSeries object.
        
        Inputs:
        -------
        series: pandas Series instance.
            Series with the historical values of the interest variable.
            It is supposed to have a monthly frequency. It is expected
            that the index of the series is a date index.
        """
        # Define attributes of the object
        self.series = series
        if isinstance(color_dict, type(None)):
            color_dict = {
                'hist': '#c00000',
                'mean': '#c00000',
                'min': '#70ad47',
                'max': '#70ad47',
                'perc_95': '#5e7493',
                'perc_5': '#5e7493',
                'hist_min': '#007179',
                'hist_max': '#007179'
            }
        self.COLORS = color_dict

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

    def plot_full_series(self, start, end, historical, 
                         figsize=(15,10), u_space=1, d_space=-1,
                         month_space=2):
        """Plots the historical and forecasted of the analized seires.
        
        Inputs:
        -------
        start: string (expected format: '%Y-%m-%d')
            String with the begining date of the plot.
        end: string (expected format: '%Y-%m-%d')
            String with the end date of the plot.
        historical: Pandas series.
            Historical observed behavior of the series. It is expected
            that the index of the series is a date index.
        forecasts: Pandas DataFrame.
            Forecast dataframe. It is expected that the index of the
            dataframe is a date index.
        figsize: tuple
            Tuple that determines the size of the figure generated by
            this method.
        u_space: numerical value
            Space between the plotted line and the label for upper
            values.
        d_space: numerical value
            Space between the plotted line and the label for lower 
            values.
        
        Outputs:
        --------
        """
        date_max = historical.idxmax()
        max_value = historical.max()
        date_min = historical.idxmin()
        min_value = historical.min()

        # Prepare data to be plotted:
        forecast = self.sim_df.agg(['min','max','mean', quant_5,
                                    quant_95], axis=1)
        forecast['Fecha'] = forecast.index
        forecast.reset_index(drop=True, inplace=True)
        series = self.series.to_frame()
        series_name = series.columns.item()
        series['Fecha'] = series.index
        series.reset_index(drop=True, inplace=True)
        temp = series.merge(forecast, how='outer', on='Fecha')
        
        # Delimit the beginning and end:
        temp = temp[(temp['Fecha']>=start) & (temp['Fecha']<=end)]
        # Plot the data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(temp['Fecha'], temp[series_name], 
                 color=self.COLORS['hist'])
        ax.plot(
            temp['Fecha'], 
            temp['quant_5'], 
            color = self.COLORS['perc_5'],
            linestyle = ':',
            linewidth = 2,
            label = 'Perc 5 - Perc 95'
        )
        ax.plot(
            temp['Fecha'], 
            temp['quant_95'], 
            color = self.COLORS['perc_95'],
            linestyle = ':',
            linewidth = 2
        )
        ax.plot(
            temp['Fecha'], 
            temp['min'], 
            color = self.COLORS['min'],
            linestyle = '--',
            linewidth = 2,
            label = 'Min - Max'
        )
        ax.plot(
            temp['Fecha'], 
            temp['max'], 
            color = self.COLORS['max'],
            linestyle = '--',
            linewidth = 2
        )
        ax.plot(
            temp['Fecha'], 
            temp['mean'], 
            color = self.COLORS['mean'],
            linewidth = 2,
            label = 'Media'
        )
        # Set the axis values:
        month_fmt = mdates.MonthLocator(interval=3)
        ax.xaxis.set_major_locator(month_fmt)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        plt.xticks(rotation=90, ha='right')

        # Give label of the last observed value:
        plt.text(
            x = self.series.index.max()-relativedelta(months=month_space+1),
            y = self.series[self.series.index.max()]+d_space,
            s = round(self.series[self.series.index.max()],1),
            color = self.COLORS['hist']
        )
        for date in forecast.loc[forecast['Fecha']<=end,'Fecha']:
            if date.month%6==0:
                
                temp_min = round(temp.loc[temp['Fecha']==date, 'min'].item(),1)
                temp_max = round(temp.loc[temp['Fecha']==date, 'max'].item(),1)
                temp_mean = round(temp.loc[temp['Fecha']==date, 'mean'].item(),1)
                temp_95 = round(temp.loc[temp['Fecha']==date, 'quant_95'].item(),1)
                temp_5 = round(temp.loc[temp['Fecha']==date, 'quant_5'].item(),1)
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = temp_min+d_space,
                    s = temp_min,
                    color = self.COLORS['min']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = temp_max+u_space,
                    s = temp_max,
                    color = self.COLORS['max']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = temp_mean+0.3*u_space,
                    s = temp_mean,
                    color = self.COLORS['mean']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = temp_95+u_space*0.7,
                    s = temp_95,
                    color = self.COLORS['perc_95']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = temp_5+d_space,
                    s = temp_5,
                    color = self.COLORS['perc_5']
                )
        plt.ylim(
            bottom = temp.select_dtypes(include='float64').min().min()\
                +1.5*d_space, 
            top = temp.select_dtypes(include='float64').max().max()\
                +1.5*u_space
        )
        plt.xlim(
            left = temp['Fecha'].min(), 
            right = temp['Fecha'].max()+relativedelta(months=month_space)
        )
        plt.legend()
        return fig, ax


#-----------------------3. Auxiliary Functions -------------------------
def quant_5(x):
    return x.quantile(0.05)

def quant_95(x):
    return x.quantile(0.95)



