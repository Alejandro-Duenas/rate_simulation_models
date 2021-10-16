#--------------------- Stochastic Modeling Module-----------------------
'''
This module has classes and functions related to the stochastic modeling
of stochastic processes. In particular, the main classes are related to
the Geometric Brownian Motion Monte Carlo simulation.
'''

# ------------------------1. Libraries----------------------------------
from datetime import timedelta
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
    def __init__(self, series, Np=1000, Nt=60, T=5, color_dict=None, 
                 u_bound=None, l_bound=None):
        """
        Inputs:
        -------
        series: pandas Series
            Series with the historical values of the variable of 
            interest. It is expected that the index of the series is a 
            date index.
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
            Nt = self.Nt,
            u_bound = self.u_bound,
            l_bound = self.l_bound
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

    def plot_full_series(self, start, end, figsize=(15,10), month_space=2,
                         dec=1):
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
                    y = pos_values['quant_5']*dist_d,
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



#-----------------------3. Auxiliary Functions -------------------------
def quant_5(x):
    return x.quantile(0.05)

def quant_95(x):
    return x.quantile(0.95)
def prop_label(x):
    if x<0: return 1.3*x
    else: return 1.05*x

