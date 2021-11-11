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
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
from dateutil.relativedelta import relativedelta

#-------------------------2. Classes------------------------------------
class GBM(object):
    '''This objects simulates a stochastic process as a Geometric 
    Brownian Motion.
    '''
    def __init__(self, s, mu, sigma, Np=1000, T=5, Nt=60,u_bound=None,
                 l_bound=None):
        """
        Inputs:
        -------
        s: numerical value
            Initial value of the Geometric Brownian Motions, from which
            the rest of the process will be simulated. It is usually the
            last observed value.
        mu: numerical value
            Mean or shift of the GBM stochastic process.
        sigma: numerical value
            Standard deviation or volatility of the stochastic process.
        Np: int
            Number of paths that will be simulated.
        T: numerical value 
            Number of periods (based on the time unit selected) that 
            will be simulated.
        Nt: integer
            Number of steps taken during T.
        u_bound: numerical value
            Upper bound to the simulated paths. If a value in the 
            simulated paths take a value greater, it will be truncated
            to this value.
        l_bound: numerical value
            Lower bound to the simulated paths. If a value in the 
            simulated paths take a value lower, it will be truncated to
            this value.

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
        self.u_bound = u_bound
        self.l_bould = l_bound
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
        
        if self.u_bound:
            s_sims = np.where(s_sims>self.u_bound, self.u_bound, s_sims)
        elif self.l_bould:
            s_sims = np.where(s_sims<self.l_bould, self.l_bould, s_sims)
        return s_sims
    
class GBMRateSeries(object):
    """This object stores a rate series, its statistical information
    and with it generates GBM simulations of the rate series. It also 
    has methods to plot historical behaviour plus its variations.
    """
    def __init__(self, series, Np=1000, Nt=60, T=60, color_dict=None, 
                 u_bound=None, l_bound=None):
        """
        Inputs:
        -------
        series: pandas Series
            Series with the historical values of the variable of 
            interest. It is expected that the index of the series is a 
            date index

        Np: integer
            Number of paths simulated

        Nt: numerical value
            Number of time steps taken along the simulation

        T: numerical value
            Time horizon over which the simulation will occur

        color_dict: dicitonary
            Dictionary that map the colors to the ploted lines in the
            visualization methods. It should contain the following keys:

            - hist: historical data line
            - mean: mean of the simulated paths
            - min: minimum value of each simulated step
            - max: maximum value of each simulated step
            - perc_95: 95th percentile of each simulated step
            - perc_5: 5th percentile of each simulated step
            - hist_min: minimum historical value
            - hist_max: maximum historical value
        
        u_bound: numerical value
            Upper bound of the simulated paths
        
        l_bound: numerical value 
            Lower bound of the simulated paths
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
        self.u_bound = u_bound
        self.l_bound = l_bound

        # Rate simulation:
        sim_df = self.simulate_rate()
        sim_df.loc[self.series.index[-1],:] = self.last
        self.sim_df = sim_df.sort_index()
    def __str__(self):
        return f'GBM {self.series.name}| Np = {self.Np}| Nt = {self.Nt}'

    def simulate_rate(self):
        """Simulates the rate according to the model inside the object.
        """
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
            Nt = self.Nt,
            u_bound = self.u_bound,
            l_bound = self.l_bound
        )
        self.dt = rate_gbm.dt
        sim_paths = rate_gbm.s_sims
        sim_df = pd.DataFrame(data=sim_paths, index=sim_date_index)
        return sim_df
    
    def get_simulated_df(self):
        return self.sim_df
    
    def get_full_df(self):
        """Returns a dataframe with Np columns with the historical and
        simulated data.
        """
        full_observed_df = pd.DataFrame(
            data = np.tile(self.series, (self.Np,1)).T,
            index = self.series.index
        )
        full_df = pd.concat([full_observed_df, self.sim_df])
        return full_df

    def plot_full_series(self, start, end, figsize=(15,10), month_space=2,
                         dec=1):
        """Plots the historical and forecasted values of the analized 
        series, with the mean, min, max, 5th and 95th percentiles of the
        simulated steps.
        
        Inputs:
        -------
        start: string (expected format: '%Y-%m-%d')
            String with the begining date of the plot.
        end: string (expected format: '%Y-%m-%d')
            String with the end date of the plot.
        figsize: tuple
            Tuple that determines the size of the figure generated by
            this method.
        month_space: int 
            determines how many time units along the x-axis the labels
            will be possitioned.
        dec: int
            Number of decimals showed in the graph. 
        
        Outputs:
        --------
        fig: matplotlib Figure
            Figure that contains the plot
        ax: matplotlib Axe
            Axe in which the plot was plotted
        """

        # Prepare data to be plotted:
        forecast = self.sim_df.agg(['min','max','mean', quant_5,
                                    quant_95], axis=1)
        forecast['Fecha'] = forecast.index
        forecast.reset_index(drop=True, inplace=True)
        series = self.series.to_frame()
        last_date = series.index.max()
        last_val = series.loc[last_date].item()
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
        d_adj = month_space+2
        local_int = temp.loc[
            (temp['Fecha']>=last_date-relativedelta(months=d_adj)) &
            (temp['Fecha']<=last_date+relativedelta(months=int(1.5*d_adj))),
            series_name
        ]
        plt.text(
            x = last_date-relativedelta(months=d_adj),
            y = local_int.max(),
            s = round(last_val,dec),
            color = self.COLORS['hist']
        )

        dates = temp.loc[temp['Fecha']>last_date, 'Fecha']
        dist_u = 1.01
        dist_d = 0.8
        for date in dates:
            if date.month%6 ==0:
                temp_min = round(temp.loc[temp['Fecha']==date, 'min'].item(),dec)
                temp_max = round(temp.loc[temp['Fecha']==date, 'max'].item(),dec)
                temp_mean = round(temp.loc[temp['Fecha']==date, 'mean'].item(),dec)
                temp_95 = round(temp.loc[temp['Fecha']==date, 'quant_95'].item(),dec)
                temp_5 = round(temp.loc[temp['Fecha']==date, 'quant_5'].item(),dec)
                local_int = temp.loc[
                    (temp['Fecha']>=date-relativedelta(months=month_space)) &
                    (temp['Fecha']<=date+relativedelta(months=int(1.5*month_space)))
                ]
                pos_values = local_int.max()
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['min']*dist_d,
                    s = temp_min,
                    color = self.COLORS['min']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['max']*dist_u,
                    s = temp_max,
                    color = self.COLORS['max']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['mean']*dist_u,
                    s = temp_mean,
                    color = self.COLORS['mean']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['quant_95']*dist_u,
                    s = temp_95,
                    color = self.COLORS['perc_95']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['quant_5']*dist_d*1.1,
                    s = temp_5,
                    color = self.COLORS['perc_5']
                )
        bottom_val = temp.select_dtypes(include='float64').min().min()
        top_val = temp.select_dtypes(include='float64').max().max()
        if bottom_val<0:
            bottom_val = bottom_val*1.1
        else:
            bottom_val = bottom_val*0.8
        plt.ylim(
            bottom = bottom_val, 
            top = top_val*1.1
        )
        plt.xlim(
            left = temp['Fecha'].min(), 
            right = temp['Fecha'].max()+relativedelta(months=month_space)
        )
        plt.legend()
        return fig, ax
    
    def plot_historic_variation(self, start='2000-01', ref=1, 
                                figsize=(15,10), month_space=2, dec=2):

        """Plots the historical variations, according to the reference 
        point given.
        
        Inputs:
        -------
        start: string (expected format: '%Y-%m-%d')
            String with the begining date of the plot.
        ref: integer
            Number of time periods in reference to which the variations
            will be computed. 
        figsize: tuple
            Tuple that determines the size of the figure generated by
            this method.
        month_space: int 
            determines how many time units along the x-axis the labels
            will be possitioned.
        dec: int
            Number of decimals showed in the graph. 
        
        Outputs:
        --------
        fig: matplotlib Figure
            Figure that contains the plot
        ax: matplotlib Axe
            Axe in which the plot was plotted
        """
        series = (self.series-self.series.shift(ref))[start:]
        hist_max = series.max()
        max_date = series.idxmax()
        hist_min = series.min()
        min_date = series.idxmin()
        last_date = series.index.max()
        start_date = series.index.min()
        series_name = series.name
        series = series.to_frame()
        series['Fecha'] = series.index
        series.reset_index(drop=True, inplace=True)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            series['Fecha'], 
            series[series_name],
            color = self.COLORS['hist'],
            label = 'Variación'
        )
        ax.axhline(hist_max, color=self.COLORS['hist_max'], 
                   linestyle=':', linewidth=2, label='Min - Max')
        ax.axhline(hist_min, color=self.COLORS['hist_min'], 
                   linestyle=':', linewidth=2)
        month_fmt = mdates.MonthLocator(interval=12)
        ax.xaxis.set_major_locator(month_fmt)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        plt.xticks(rotation=90, ha='right')
        plt.text(
            x = start_date-relativedelta(months=month_space),
            y = hist_max*1.01, 
            s = f"{max_date.strftime('%b-%y')}: {round(hist_max,dec)}",
            color = self.COLORS['hist_max']
        )
        plt.text(
            x = start_date-relativedelta(months=month_space),
            y = hist_min*1.1, 
            s = f"{min_date.strftime('%b-%y')}: {round(hist_min,dec)}",
            color = self.COLORS['hist_min']
        )
        for date in series['Fecha']:
            if date.month==last_date.month:
                local_int = series.loc[
                    (series['Fecha']>=date-relativedelta(months=month_space)) &
                    (series['Fecha']<=date+relativedelta(months=2*month_space)),
                    series_name
                    ]
                val = series.loc[series['Fecha']==date, series_name].item()
                if val>0 :
                    plt.text(
                        x = date-relativedelta(months=month_space),
                        y = local_int.max()*1.05,
                        s = round(val, dec),
                        color = self.COLORS['hist']
                    )
                else:
                    plt.text(
                        x = date-relativedelta(months=month_space),
                        y = local_int.min()*1.1,
                        s = round(val, dec),
                        color = self.COLORS['hist']
                    )

        plt.xlim(
            left = start_date-relativedelta(months=month_space),
            right = last_date+relativedelta(months=month_space)
        )
        plt.ylim(
            bottom = hist_min*1.2,
            top = hist_max*1.2
        )
        plt.legend()
        return fig, ax

    def plot_full_variations(self, start, end, ref=1, figsize=(15,10), 
                             month_space=2, dec=2):
        """Plots the historical and forecasted variations of the analized 
        series, with the mean, min, max, 5th and 95th percentiles of the
        simulated steps.
        
        Inputs:
        -------
        start: string (expected format: '%Y-%m-%d')
            String with the begining date of the plot.
        end: string (expected format: '%Y-%m-%d')
            String with the end date of the plot.
        ref: integer
            Number of time periods in reference to which the variations
            will be computed.
        figsize: tuple
            Tuple that determines the size of the figure generated by
            this method.
        month_space: int 
            determines how many time units along the x-axis the labels
            will be possitioned.
        dec: int
            Number of decimals showed in the graph. 
        
        Outputs:
        --------
        fig: matplotlib Figure
            Figure that contains the plot
        ax: matplotlib Axe
            Axe in which the plot was plotted
        """
        # Get variations:
        var_series = (self.series-self.series.shift(ref)).dropna()
        # var_series = np.log(self.series/self.series.shift(ref))
        series_name = var_series.name
        var_forecast = self.get_full_df()-self.get_full_df().shift(ref)
        # var_forecast = np.log(self.get_full_df()/self.get_full_df().shift(ref))
        var_forecast = var_forecast.loc[var_series.index.max():]
        var_forecast =  var_forecast.agg(
            ['min','max','mean',quant_5, quant_95],
            axis = 1
        )

        # Get important variables
        var_series = var_series[start:]
        hist_max = var_series.max()
        max_date = var_series.idxmax()
        hist_min = var_series.min()
        min_date = var_series.idxmin()
        last_date = var_series.index[-1]
        start_date = var_series.index[0]
        last_obs_val = var_series.last('1D').item()

        # Give the correct format and join the full information of variations:
        var_series = var_series.to_frame()
        var_series['Fecha'] = var_series.index
        var_series.reset_index(drop=True, inplace=True)
        var_forecast['Fecha'] = var_forecast.index
        var_forecast.reset_index(drop=True, inplace=True)
        full_var = var_series.merge(var_forecast, how='outer', on='Fecha')

        # Delimit the data's beginning and end:
        full_var = full_var[
            (full_var['Fecha']>=start) &
            (full_var['Fecha']<=end)
        ]

        # Plot the data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(full_var['Fecha'], full_var[series_name],
                color = self.COLORS['hist'])
        ax.plot(
            full_var['Fecha'], 
            full_var['quant_5'], 
            color = self.COLORS['perc_5'],
            linestyle = ':',
            linewidth = 2,
            label = 'Perc 5 - Perc 95'
        )
        ax.plot(
            full_var['Fecha'], 
            full_var['quant_95'], 
            color = self.COLORS['perc_95'],
            linestyle = ':',
            linewidth = 2
        )
        ax.plot(
            full_var['Fecha'], 
            full_var['min'], 
            color = self.COLORS['min'],
            linestyle = '--',
            linewidth = 2,
            label = 'Min - Max'
        )
        ax.plot(
            full_var['Fecha'], 
            full_var['max'], 
            color = self.COLORS['max'],
            linestyle = '--',
            linewidth = 2
        )
        ax.plot(
            full_var['Fecha'], 
            full_var['mean'], 
            color = self.COLORS['mean'],
            linewidth = 2,
            label = 'Media'
        )
        ax.axhline(hist_max, color=self.COLORS['hist_max'], 
                   linestyle=':', linewidth=2, label='Min - Max')
        ax.axhline(hist_min, color=self.COLORS['hist_min'], 
                   linestyle=':', linewidth=2)
        
        # Set the axis values:
        month_fmt = mdates.MonthLocator(interval=3)
        ax.xaxis.set_major_locator(month_fmt)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        plt.xticks(rotation=90, ha='right')

        # Give label of the reference values:
        da = month_space+1
        local_int = full_var.loc[
            (full_var['Fecha']>=last_date-relativedelta(months=da)) &
            (full_var['Fecha']<=last_date+relativedelta(months=int(1.5*da))),
            series_name
        ]
        plt.text(
            x = last_date-relativedelta(months=da),
            y = 1.05*local_int.max(),
            s = round(last_obs_val,dec),
            color = self.COLORS['hist']
        )
        plt.text(
            x = start_date,
            y = prop_label(hist_max), 
            s = f"{max_date.strftime('%b-%y')}: {round(hist_max,dec)}",
            color = self.COLORS['hist_max']
        )
        plt.text(
            x = start_date,
            y = prop_label(hist_min), 
            s = f"{min_date.strftime('%b-%y')}: {round(hist_min,dec)}",
            color = self.COLORS['hist_min']
        )

        dates = full_var.loc[full_var['Fecha']>last_date, 'Fecha']
        dist_u = 1.01
        dist_d = 0.96
        for date in dates:
            if date.month%6 == 0:
                temp_min = round(
                    full_var.loc[full_var['Fecha']==date, 'min'].item(),
                    dec
                )
                temp_max = round(
                    full_var.loc[full_var['Fecha']==date, 'max'].item(),
                    dec
                )
                temp_mean = round(
                    full_var.loc[full_var['Fecha']==date, 'mean'].item(),
                    dec
                )
                temp_95 = round(
                    full_var.loc[full_var['Fecha']==date, 'quant_95'].item(),
                    dec
                )
                temp_5 = round(
                    full_var.loc[full_var['Fecha']==date, 'quant_5'].item(),
                    dec
                )
                local_int = full_var.loc[
                    (full_var['Fecha']>=date-relativedelta(months=month_space)) &
                    (full_var['Fecha']<=date+relativedelta(months=int(1.5*month_space)))
                ]
                pos_values = local_int.max()
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['min']*dist_d,
                    s = temp_min,
                    color = self.COLORS['min']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['max']*dist_u,
                    s = temp_max,
                    color = self.COLORS['max']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['mean']*dist_u,
                    s = temp_mean,
                    color = self.COLORS['mean']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['quant_95']*dist_u,
                    s = temp_95,
                    color = self.COLORS['perc_95']
                )
                plt.text(
                    x = date-relativedelta(months=month_space), 
                    y = pos_values['quant_5']*dist_d,
                    s = temp_5,
                    color = self.COLORS['perc_5']
                )
        bottom_val = full_var.select_dtypes(include='float64').min().min()
        top_val = full_var.select_dtypes(include='float64').max().max()
        if bottom_val<0:
            bottom_val = bottom_val*1.1
        else:
            bottom_val = bottom_val*0.9
        plt.ylim(
            bottom = bottom_val, 
            top = top_val*1.1
        )
        plt.xlim(
            left = full_var['Fecha'].min(), 
            right = full_var['Fecha'].max()+relativedelta(months=month_space)
        )
        plt.legend()
        return fig, ax

class PoissonRateSeries(GBMRateSeries):
    def __init__(self, series, Np=1000, Nt=60, T=60, color_dict=None, 
        u_bound=None, l_bound=None, ref_date='2011-01-01',p=2, q=4, i=0):
        """
        Inputs:
        -------
        series: pandas Series
            Series with the historical values of the variable of 
            interest. It is expected that the index of the series is a 
            date index

        Np: integer
            Number of paths simulated

        Nt: numerical value
            Number of time steps taken along the simulation

        T: numerical value
            Time horizon over which the simulation will occur

        color_dict: dicitonary
            Dictionary that map the colors to the ploted lines in the
            visualization methods. It should contain the following keys:

            - hist: historical data line
            - mean: mean of the simulated paths
            - min: minimum value of each simulated step
            - max: maximum value of each simulated step
            - perc_95: 95th percentile of each simulated step
            - perc_5: 5th percentile of each simulated step
            - hist_min: minimum historical value
            - hist_max: maximum historical value
        
        u_bound: numerical value
            Upper bound of the simulated paths
        
        l_bound: numerical value 
            Lower bound of the simulated paths
        
        ref_date: str (expected format '%Y-%m-%d')
            String with the initial date of analysis to compute the 
            parameters of the model.
        
        p: int
            Number of AR lags for the ARMA model.
        
        q: int
            Number of MA lags for the ARIMA model.
        
        i: int
            Number of I differentiations for the ARIMA model
        """
        super().__init__(self, series, Np, Nt, T, color_dict, u_bound,
            l_bound)
        self.p = p
        self.q = q
        self.i = i
        self.jump_series = series.loc[series/series.shift(1)!=1]
        self.lmbda = self.compute_lambda()
        self.jump_series = self.jump_series[ref_date:]

    def compute_lambda(self):
        temp_series = self.jump_series.copy().to_frame()
        temp_series['Fecha'] = temp_series.index
        temp_series.reset_index(drop=True, inplace=True)
        cd = temp_series['Fecha']
        shift_cd = cd.shift(1)
        temp_series['delta_t'] = 12*(cd.dt.year-shift_cd.dt.year)+cd.dt.month-\
            shift_cd.dt.month
        temp_series = temp_series.loc[temp_series['Fecha']>='2011-01-01']
        temp_series['class'] = temp_series['delta_t'].apply(jump_class)
        lmbda = temp_series['class'].mean()
        return lmbda

    def simulate_rate(self):
        trans_series = -np.log(0.15/self.jump_series-1)
        model = ARIMA(trans_series, order=(self.p, self.i, self.q))
        fitted = model.fit()
        self.arima_summary = fitted.summary()
        res_std = fitted.resid.std()
        trans_forecast = fitted.forecast(60).reshape(60,1)
        mc_trans_forecast = trans_forecast+np.random.normal(
            scale = res_std*np.sqrt(12),
            size = (60,1000)
        )
        mc_forecast = 0.15/(1+np.exp(-mc_trans_forecast))
        jump_sim = (0-np.log(np.random.uniform(size=(60,1000)))/self.lmbda)\
            .round(0).cumsum(axis=0)
        row_index = np.where(jump_sim>=60, np.NaN, jump_sim)
        row_index = pd.DataFrame(row_index).fillna(method='ffill').values
        
        mc_final = np.zeros((60,1000))
        for i in range(1000):
            mc_final[:,i] = mc_forecast[row_index[:,i],i]
        mc_final

        



#-----------------------3. Auxiliary Functions -------------------------
def quant_5(x):
    '''Returns the 5th percentile.
    Inputs:
    -------
    x: array-like
    '''
    return x.quantile(0.05)

def quant_95(x):
    '''Returns the 95th percentile.
    Inputs:
    -------
    x: array-like
    '''
    return x.quantile(0.95)
def prop_label(x):
    '''Returns a proportion of value x. It is used to possition labels
    in plots.
    '''
    if x<0: return 1.3*x
    else: return 1.05*x
def jump_class(x):
    if x >= 5: return 5
    elif x == 0: return 1
    else: return x

