#--------------------- Stochastic Modeling Module-----------------------
'''
This module has classes and functions related to the stochastic modeling
of stochastic processes. In particular, the main classes are related to
the Geometric Brownian Motion Monte Carlo simulation.
'''

# ------------------------1. Libraries----------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.morestats import Mean
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from statsmodels.tsa.arima.model import ARIMA
from dateutil.relativedelta import relativedelta
import datetime
from typing import Union
from .utils import *

#-------------------------2. Classes------------------------------------

# 1. Geometric Brownian Motion process class:

class GBM(object):
    """This objects simulates a stochastic process as a Geometric 
    Brownian Motion
    """

    def __init__(
            self, s: float, mu:float, sigma:float, Np:int=1000, 
            T:int=5, Nt:int=60, upper_bound:float=None, 
            lower_bound:float=None
            ):
        """
        Args:
            s (float): initial value of the Geometric Brownian Motion,
                from which the rest of the process will be simulated. It
                is usually the last observed value.
            mu (float): mean or shift of the GBM.
            sigma (float): stadard devation or volatility of the GBM.
            Np (int, optional): number of paths that will be simulated. 
                Defaults to 1000.
            T (int, optional): number of periods (based on the time unit
                selected) that will be simulated. Defaults to 5.
            Nt (int, optional): number of steps taken during T. Defaults
                to 60.
            u_bound (float, optional): upper bound to the simulated 
                paths. If the value in the simulated paths take a 
                greater value, it will be truncated to this value. 
                Defaults to None.
            l_bound (float, optional): lower bound to the simulated 
                paths. If the value in the simulated paths take a lower 
                value, it will be truncated to this value. Defaults to 
                None.
        """

        self.s = s
        self.mu = mu
        self.sigma = sigma
        self.Np = Np
        self.T = T
        self.Nt = Nt
        self.dt = T/Nt
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    
    @property
    def simulated_paths(self):
        """Simulated Np paths of a Geometric Brownian Motion stochastic
        process. This simulation follows the logarithmic transformation
        on the GBM.
        """
        brownian_motion = np.random.normal(
            scale = self.dt**0.5,
            size = (self.Nt, self.Np)
        )
        first_term = (self.mu-self.sigma**2/2)*self.dt
        simulated_paths = self.s*np.exp(
            (first_term+self.sigma*brownian_motion).cumsum(axis=0))
        
        # Bound the simulation paths:
        if self.upper_bound:
            simulated_paths = np.where(
                simulated_paths>self.upper_bound, 
                self.upper_bound, 
                simulated_paths
                )
        elif self.lower_bound:
            simulated_paths = np.where(
                simulated_paths<self.lower_bound, 
                self.lower_bound, 
                simulated_paths
                )

        return simulated_paths


# 2. Parent RateSeriesSimulation class:

class RateSeriesSimulation(object):
    """This is a general object used as parent class for the rate 
    modeling subclasses as GBMRateSeries. It containes methods to model 
    and plot time series simulations.
    """

    def __init__(
        self, series: Union[pd.DataFrame, pd.Series], Np: int=1000, 
        Nt: int=60, T: int=60, color_dict: dict=None,
        upper_bound: float=None, lower_bound: float=None
        ):
        """
        Args:
            series (Union[pd.DataFrame, pd.Series]): contains the data 
                of the series that will be modeled.
            Np (int, optional): number of paths simlated. Defaults to 
                1000.
            Nt (int, optional): number of time steps simulated. Defaults
                to 60.
            T (int, optional): time horizon over which the simulation is
                done. Defaults to 60.
            color_dict (dict, optional): dictionary that maps the colors
                to the plotted lines in the visualization methods. It 
                should contain the following keys:

                - hist: historical data line
                - mean: mean of the simulated paths
                - min: minimum value of each simulated step
                - max: maximum value of each simulated step
                - perc_95: 95th percentile of each simulated step
                - perc_5: 5th percentile of each simulated step
                - hist_min: minimum historical value
                - hist_max: maximum historical value
                
                Defaults to None.
            upper_bound (float, optional): upper bound value for the
                simulations.
            lower_bound (float, optional): lower bound value for the
                simulations.

        Returns:
            RateSeriesSimulation: instance of the RateSeriesSimulation
                class.
        """
        # Define variables:
        if not color_dict:
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

        # Define attributes:
        self.series = series
        self.COLORS = color_dict
        self.Np= Np
        self.Nt = Nt
        self.T = T
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.simulated_df = self.simulate_df()
    
    def simulate_df(self)->pd.DataFrame:
        """"With the attributes and methods of the object, simulates the
        interest rate.

        Returns:
            pd.DataFrame: DataFrame with the simulated seires
        """
        return pd.DataFrame()
    
    @property
    def full_df(self)->pd.DataFrame:
        """Generates a pandas DataFrame with Np columns with the 
        historical and simulated data concatenated. For the historical
        data, it repeats the values each column. For the simulated data
        each column is a different simulated path.

        Returns:
            pd.DataFrame: dataframe with the historical and simulated
                data.
        """
        observed_df = pd.DataFrame(
            data = np.tile(self.series, (self.Np, 1)).T,
            index = self.series.index
            )
        full_df = pd.concat([observed_df, self.simulated_df])
        return full_df

    @staticmethod
    def aggregate_simulations(df: pd.DataFrame, axis: int=1)-> pd.DataFrame:
        """Aggregates into minimum, maximum, mean, 5th and 95th 
        percentiles, from axis. 

        Args:
            df (pd.DataFrame): contains the simulation data that will be
                aggregated.
            axis (int): axis in which the aggregation is done.

        Returns:
            pd.DataFrame: 5 column/row pd.DataFrame with the aggregated 
                data.
        """
        agg_df = df.agg(
            func = ['min', 'max', 'mean', quant_5, quant_95],
            axis = axis
            )

        return agg_df
        
        
    def variation_series(self, delta_t: int=1)->Union[pd.DataFrame, pd.Series]:
        """Computes the attribute of variation series from the series of
        interest.

        Args:
            delt_t (int): time delta from which the variations are
                computed. Defaults to 1.
                
        Returns:
            Union[pd.DataFrame, pd.Series]: pandas object with the 
                series of variations
        """
        variation_series = (self.series-self.series.shift(delta_t)).dropna()
        return variation_series
    
    def variation_forecast_df(self, delta_t: int=1)->pd.DataFrame:
        """Computes the variation of all paths for the forecasts in
        simulated_df attribute.

        Args:
            delta_t (int, optional): time delta from which the 
                variations are computed. Defaults to 1.

        Returns:
            pd.DataFrame: dataframe with the variations of each Np paths
                for the time delta inputed.
        """
        var_forecast = self.full_df-self.full_df.shift(delta_t)
        init_date = self.series.index.max()
        var_forecast = var_forecast.loc[init_date:]

        return var_forecast
    
    

    def plot_full_series(
        self, start: Union[str, datetime.datetime], 
        end: Union[str, datetime.datetime], figsize: tuple=(15,10),
        time_space: float=2., dec: int=1,
        upper_space: int=0.1, lower_space: float=0.1
        ):
        """Plots the historical and forecasted values of the analized 
        series, with the mean, min, max, 5th and 95th percentiles of the
        simulated steps.

        Args:
            start (Union[str, datetime.datetime]): strin with the
                beginning date of the plot. Expected format: '%Y-%m-%d'.
            end (Union[str, datetime.datetime]): strin with the 
                beginning date of the plot. Expected format: '%Y-%m-%d'.
            figsize (tuple, optional): [description]. Defaults to 
                (15,10).
            time_space (float, optional): value of time units along the
                x-axis the labels will be possitioned. Defaults to 2.
            dec (int, optional): number of decimals in the labels of the
                plot. Defaults to 1.

        Returns:
            tuple: tuple with mlp.figure.Figure, 
                mlp..axes._subplots.AxesSubplot which contain the plot
                objects.
        """

        init_date = self.series.index.max()
        forecast = self.aggregate_simulations(self.simulated_df)
        forecast.loc[init_date, :] = self.series[init_date]
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
        local_int = temp.loc[
            (temp['Fecha']>=last_date-relativedelta(months=time_space)) &
            (temp['Fecha']<=last_date+relativedelta(months=int(time_space))),
            series_name
            ]
        plt.text(
            x = last_date-relativedelta(months=time_space),
            y = local_int.max(),
            s = round(last_val, dec),
            color = self.COLORS['hist']
            )

        dates = temp.loc[temp['Fecha']>last_date, 'Fecha']
        dist_u = 1+upper_space
        dist_d = 1-lower_space

        for date in dates:
            if date.month%6==0 and (date-last_date).days>90:

                # Find label's valeus:
                temp_min = round(temp.loc[temp['Fecha']==date, 'min'].item(),
                                 dec)
                temp_max = round(temp.loc[temp['Fecha']==date, 'max'].item(),
                                 dec)
                temp_mean = round(temp.loc[temp['Fecha']==date, 'mean'].item(),
                                  dec)
                temp_95 = round(
                    temp.loc[temp['Fecha']==date, 'quant_95'].item(),
                    dec
                    )
                temp_5 = round(temp.loc[temp['Fecha']==date, 'quant_5'].item(),
                               dec)

                local_int = temp.loc[
                    (temp['Fecha']>=date-relativedelta(months=time_space)) &
                    (temp['Fecha']<=date+relativedelta(months=int(time_space)))
                    ]

                pos_values = local_int.max()

                # Put labels in plot:
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['min']*dist_d,
                    s = temp_min,
                    color = self.COLORS['min']
                    )
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['max']*dist_u,
                    s = temp_max,
                    color = self.COLORS['max']
                    )
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['mean']*dist_u,
                    s = temp_mean,
                    color = self.COLORS['mean']
                    )
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['quant_95']*dist_u,
                    s = temp_95,
                    color = self.COLORS['perc_95']
                    )
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['quant_5']*dist_d*1.1,
                    s = temp_5,
                    color = self.COLORS['perc_5']
                    )

        bottom_val = temp.select_dtypes(include='float64').min().min()
        top_val = temp.select_dtypes(include='float64').max().max()

        if bottom_val<0:
            bottom_val = bottom_val*1.1
        else:
            bottom_val = bottom_val*0.9

        plt.ylim(
            bottom = bottom_val, 
            top = top_val*1.1
            )
        plt.xlim(
            left = temp['Fecha'].min(), 
            right = temp['Fecha'].max()+relativedelta(months=time_space)
            )
        plt.legend()
        plt.show()

        return fig, ax

    def plot_historic_variation(
        self, start: Union[str, datetime.datetime]='2000-01-01',
        delta_t: int=1, figsize: tuple=(15,10), time_space: int=2,
        dec: int=2, upper_space: int=0.1, lower_space: float=0.1
        ):
        """Plots the historical variations of the interest series.

        Args:
            start (str, optional): starting date of the plot. Defaults 
                to '2000-01-01'. Expected format'%Y-%m-%d'
            delta_t (int, optional): delta of time from which the 
                variation will be computed. Defaults to 1.
            figsize (tuple, optional): figure size. Defaults to (15,10).
            time_space (int, optional): number of time periods along the
                x-axis from which the labels will be positioned. 
                Defaults to 2.
            dec (int, optional): number of decimals in the labels.
                Defaults to 2.
            upper_space (int, optional): determines the position of the
                upper labels. Defaults to 0.1.
            lower_space (float, optional): determines the position of the
                lower labels. Defaults to 0.1.

        Returns:
            tuple: tuple with mlp.figure.Figure, 
                mlp..axes._subplots.AxesSubplot which contain the plot
                objects.
        """
        # Compute variation series and key values:
        series = self.variation_series(delta_t=delta_t)[start:]
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
        upper_space = upper_space+1
        lower_space = 1-lower_space

        # Plot variation series with labels:
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
            x = start_date-relativedelta(months=time_space),
            y = hist_max*upper_space, 
            s = f"{max_date.strftime('%b-%y')}: {round(hist_max, dec)}",
            color = self.COLORS['hist_max']
            )
        plt.text(
            x = start_date-relativedelta(months=time_space),
            y = hist_min*upper_space, 
            s = f"{min_date.strftime('%b-%y')}: {round(hist_min, dec)}",
            color = self.COLORS['hist_min']
            )
        for date in series['Fecha']:
            if date.month==last_date.month:
                local_int = series.loc[
                    (series['Fecha']>=date-relativedelta(months=time_space)) &
                    (series['Fecha']<=date+relativedelta(months=time_space)),
                    series_name
                    ]
                val = series.loc[series['Fecha']==date, series_name].item()
                if val>0 :
                    plt.text(
                        x = date-relativedelta(months=time_space),
                        y = local_int.max()*upper_space,
                        s = round(val, dec),
                        color = self.COLORS['hist']
                        )
                else:
                    plt.text(
                        x = date-relativedelta(months=time_space),
                        y = local_int.min()*lower_space,
                        s = round(val, dec),
                        color = self.COLORS['hist']
                        )
        if hist_min<0:
            bottom = hist_min*1.2
        else:
            bottom = hist_min*0.8

        plt.xlim(
            left = start_date-relativedelta(months=time_space),
            right = last_date+relativedelta(months=time_space)
            )
        plt.ylim(
            bottom = bottom,
            top = hist_max*1.2
            )
        plt.legend()
        plt.show()

        return fig, ax
    
    def plot_full_variations(
        self, start: Union[str, datetime.datetime],
        end: Union[str, datetime.datetime], delta_t: int=1,
        figsize: tuple=(15,10), time_space: int=2, dec: int=2,
        upper_space: int=0.1, lower_space: float=0.1
        )->tuple:
        """Plots both the historical variations and the forecasted
        variations of the series, with its, mean,  cone of confidence 
        interval and maximum and minimum.

        Args:
            start (Union[str, datetime.datetime]): start date of the
                plot. Expected format: '%Y-%m-%d'.
            end (Union[str, datetime.datetime]):  end date of the
                plot. Expected format: '%Y-%m-%d'.
            delta_t (int, optional): delta of time from which the
                variation will be computed. Defaults to 1.
            figsize (tuple, optional): figure size. Defaults to (15,10).
            time_space (int, optional): number of time periords along 
                the x-axis from which the labels will be positioned. 
                Defaults to 2.
            dec (int, optional): number of decimals in the labels. 
                Defaults to 2.
            upper_space (int, optional): determines the position of the
                upper labels. Defaults to 0.1.
            lower_space (float, optional): determines the position of 
                the lower labels. Defaults to 0.1.

        Returns:
            tuple: tuple with mlp.figure.Figure, 
                mlp..axes._subplots.AxesSubplot which contain the plot
                objects.
        """
        # Get variations:
        var_series = self.variation_series()
        series_name = var_series.name
        var_forecast = self.variation_forecast_df(delta_t=delta_t)
        var_forecast =  self.aggregate_simulations(var_forecast)

        # Get important variables
        var_series = var_series[start:]
        hist_max = var_series.max()
        max_date = var_series.idxmax()
        hist_min = var_series.min()
        min_date = var_series.idxmin()
        last_date = var_series.index[-1]
        start_date = var_series.index[0]
        last_obs_val = var_series.last('1D').item()
        dist_u = 1+upper_space
        dist_d = 1-lower_space

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
        local_int = full_var.loc[
            (full_var['Fecha']>=last_date-relativedelta(months=time_space)) &
            (full_var['Fecha']<=last_date+relativedelta(months=time_space)),
            series_name
            ]
        plt.text(
            x = last_date-relativedelta(months=time_space),
            y = dist_u*local_int.max(),
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
        for date in dates:
            if date.month%6 == 0 and (date-last_date).days>90:
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
                    (full_var['Fecha']>=date-relativedelta(months=time_space)) &
                    (full_var['Fecha']<=date+relativedelta(months=time_space))
                ]
                pos_values = local_int.max()
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['min']*dist_d,
                    s = temp_min,
                    color = self.COLORS['min']
                )
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['max']*dist_u,
                    s = temp_max,
                    color = self.COLORS['max']
                )
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['mean']*dist_u,
                    s = temp_mean,
                    color = self.COLORS['mean']
                )
                plt.text(
                    x = date-relativedelta(months=time_space), 
                    y = pos_values['quant_95']*dist_u,
                    s = temp_95,
                    color = self.COLORS['perc_95']
                )
                plt.text(
                    x = date-relativedelta(months=time_space), 
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
            right = full_var['Fecha'].max()+relativedelta(months=time_space)
        )
        plt.legend()
        return fig, ax


# 3. Child GBMRateSeries class:

class GBMRateSeries(RateSeriesSimulation):
    """This object stores a rate series, its statistical information
    and with it generates GBM simulations of the rate series. It also 
    has methods to plot historical behaviour plus its variations.
    """

    def __init__(
        self, series: Union[pd.DataFrame, pd.Series], Np: int=1000, 
        Nt: int=60, T: int=60, color_dict: dict=None, 
        upper_bound: float=None, lower_bound: float=None
        ):
        """
        Args:
            series (Union[pd.DataFrame, pd.Series]): contains the data 
                of the series that will be modeled.
            Np (int, optional): number of paths simlated. Defaults to 
                1000.
            Nt (int, optional): number of time steps simulated. Defaults
                to 60.
            T (int, optional): time horizon over which the simulation is
                done. Defaults to 60.
            color_dict (dict, optional): dictionary that maps the colors
                to the plotted lines in the visualization methods. It 
                should contain the following keys:

                - hist: historical data line
                - mean: mean of the simulated paths
                - min: minimum value of each simulated step
                - max: maximum value of each simulated step
                - perc_95: 95th percentile of each simulated step
                - perc_5: 5th percentile of each simulated step
                - hist_min: minimum historical value
                - hist_max: maximum historical value
                                                                                                
                Defaults to None.
            upper_bound (float, optional): upper bound value for the
                simulations.
            lower_bound (float, optional): lower bound value for the
                simulations.

        Returns:
            RateSeriesSimulation: instance of the RateSeriesSimulation
                class.
        """
        # Log-monthly changes:
        log_changes = np.log(series/series.shift(1)).dropna()
        self.mu = log_changes.mean()
        self.sigma = log_changes.std()
        self.last = series.last('1D').item()

        RateSeriesSimulation.__init__(
            self,
            series = series,
            Np = Np,
            Nt = Nt,
            T = T, 
            color_dict = color_dict, 
            upper_bound = upper_bound,
            lower_bound = lower_bound
            )

        

    def __str__(self):
        return f'GBM {self.series.name}| Np = {self.Np}| Nt = {self.Nt}'

    def simulate_df(self) -> pd.DataFrame:
        """Simulates the interest rate of interest following a 
        Geometric Brownian Motion process.

        Returns:
            pd.DataFrame: Np paths simulated of the rate for Nt periods
                in the future.
        """
        sim_date_index = pd.date_range(
            start = self.series.index[-1]+relativedelta(months=1),
            end = self.series.index[-1]+relativedelta(months=self.Nt),
            freq = 'M'
            )
        
        # Generate the GBM simulation:
        rate_gbm = GBM(
            s = self.last,
            mu = self.mu,
            sigma = self.sigma,
            Np = self.Np,
            T = self.T,
            Nt = self.Nt,
            upper_bound = self.upper_bound,
            lower_bound = self.lower_bound
            )

        self.dt = rate_gbm.dt
        simulated_paths = rate_gbm.simulated_paths
        simulated_df = pd.DataFrame(data=simulated_paths, index=sim_date_index)

        return simulated_df

class PoissonRateSeries(RateSeriesSimulation):
    """This object stores a rate series (expected as the policy rate of
    a central bank), its statitical information, and with it, generates
    a simulation of the rate with a combinaiton of an ARIMA and a 
    Poisson process jump model, to simulate the process by which the
    rate jumps.
    """
    def __init__(
        self, series: pd.Series, Np: int=1000, Nt: int=60, T: int=60, 
        color_dict: dict=None, 
        ref_date: Union[str, datetime.datetime]='2011-01-01',
        p: int=4, q: int=4, i: int=0):
        """
        Args:
            series (pd.Series): contains the data of the series that
                will be modeled.
            Np (int, optional): number of paths simlated. Defaults to 
                1000.
            Nt (int, optional): number of time steps simulated. Defaults
                to 60.
            T (int, optional): time horizon over which the simulation is
                done. Defaults to 60.
            color_dict (dict, optional): dictionary that maps the colors
                to the plotted lines in the visualization methods. It 
                should contain the following keys:

                - hist: historical data line
                - mean: mean of the simulated paths
                - min: minimum value of each simulated step
                - max: maximum value of each simulated step
                - perc_95: 95th percentile of each simulated step
                - perc_5: 5th percentile of each simulated step
                - hist_min: minimum historical value
                - hist_max: maximum historical value
                                                                                                
                Defaults to None.
            ref_date (Union[str, datetime.datetime], optional): initial
                date of analysis to compute the parameter of the model.
                Defaults to '2011-01-01'.
            p (int, optional): number of AR lags for the ARIMA model.
                Defaults to 4.
            q (int, optional): number of MA lags for ARIMA model. 
                Defaults to 4.
            i (int, optional): order of integration in the series for
                the ARIMA model. Defaults to 0.
        """
        
        self.p = p
        self.q = q
        self.i = i
        self.jump_series = series.loc[series/series.shift(1)!=1]
        self.jump_series = self.jump_series[ref_date:]
        RateSeriesSimulation.__init__(self, series, Np, Nt, T, color_dict)
        self.series = self.series*100
        
    def __str__(self):
        return f"ARIMA(p: {self.p}, i: {self.i}, q: {self.q}); \n"+\
            f"Poisson dist (lambda: {self.lmbda:4.2f.}"

    @property
    def lmbda(self):
        """Computes the lambda value for the Poisson process.
        """
        temp_series = self.jump_series.copy().to_frame()
        temp_series['Fecha'] = temp_series.index
        temp_series.reset_index(drop=True, inplace=True)
        cur_date = temp_series['Fecha']
        shift_cd = cur_date.shift(1)
        temp_series['delta_t'] = 12*(cur_date.dt.year-shift_cd.dt.year)+\
            cur_date.dt.month-shift_cd.dt.month
        temp_series = temp_series.loc[temp_series['Fecha']>='2011-01-01']
        
        temp_series['class'] = np.vectorize(jump_class)(
            temp_series['delta_t'].values
            )
        lmbda = temp_series['class'].mean()

        return lmbda
    
    def aggregate_simulations(self, df: pd.DataFrame, 
                              axis: int = 1) -> pd.DataFrame:
        """Aggregates into minimum, maximum, mean, 5th and 95th 
        percentiles, from axis. It also approximates to the .25 closer
        value. 

        Args:
            df (pd.DataFrame): contains the simulation data that will be
                aggregated.
            axis (int): axis in which the aggregation is done.

        Returns:
            pd.DataFrame: 5 column/row pd.DataFrame with the aggregated 
                data.
        """
        agg_df = RateSeriesSimulation.aggregate_simulations(df, axis=axis)
        agg_df = np.round(agg_df*4, 0)/4
        return agg_df

    def simulate_df(self)->pd.DataFrame:
        """With the historic behavior of the interest rate, and the
        computed lambda value that characterizes a Poisson process, it 
        simulates the rate with a combination of an ARIMA model and a
        Poisson process.

        Returns:
            pd.DataFrame: (Nt, Np) sized DataFrame with the Np simulated
                paths.
        """
        # Generate the date index of the simulation:
        sim_date_index = pd.date_range(
            start = self.series.index[-1]+relativedelta(months=1),
            end = self.series.index[-1]+relativedelta(months=self.Nt),
            freq = 'M'
            )
        # Generate an inverse sigmoid transformation of the data:
        trans_series = -np.log(0.15/self.jump_series-1).values

        # Generate the ARIMA simulation for the rate:
        model = ARIMA(trans_series, order=(self.p, self.i, self.q))
        fitted = model.fit()
        self.arima_summary = fitted.summary()
        res_std = fitted.resid.std()
        trans_forecast = fitted.forecast(60).reshape(60,1)

        # Generate different paths from the ARIMA model:
        mc_trans_forecast = trans_forecast+np.random.normal(
            scale = res_std*np.sqrt(1),
            size = (60,1000)
            )

        # Bring back to % units the forecast:
        mc_forecast = 0.15/(1+np.exp(-mc_trans_forecast))

        # Determine the jumps in time according to Poisson distribution:
        jump_sim = (0-np.log(np.random.uniform(size=(60,1000)))/self.lmbda)\
            .round(0).cumsum(axis=0)
        row_index = np.where(jump_sim>=60, np.NaN, jump_sim)
        row_index = pd.DataFrame(row_index).fillna(method='ffill').values.\
            astype(int)
        mc_final = np.zeros((60,1000))
        
        for i in range(1000):
            mc_final[:,i] = mc_forecast[row_index[:,i],i]
        
        simulated_df = pd.DataFrame(data=mc_final, index=sim_date_index)

        # Round to closer 0.25 multiple, with two decimals:
        simulated_df = np.round(simulated_df*40000, 0)/400

        return simulated_df
    
        
class VARModelSeries(GBMRateSeries):
    """This model contains bi-varible VAR model methods and attributes
    for the forecasting of the two variables.
    """
    def __init__(self, df:pd.DataFrame, 
        Np: int = 1000,
        Nt: int = 60, 
        T: int = 60,
        color_dict: Union[dict, None] = None,
        ref_date: str = '2011-01-01', 
        lags: Union[tuple, list] = ((1,1),(2,2)), 
        trans_par: Union[list, tuple, None] = (1, 0.15)):
        """
        Inputs:
        -------
        df: pandas Data Frame
            Data frame with the historical behavior of the interest variables.
            It is expected that the index is a datetime index. It is supposed
            that the TIBR rate is the first column.
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
        
        lags: List/tuple
            Contains the lags used for each variable in the VAR model.
        
        trans_par: 
        """
        self.lags = lags
        self.df = df
        self.Np = Np
        self.Nt = Nt
        self.T = T
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
        self.trans_par = trans_par
    
    def __str__(self):
        return f'VAR Moder ({list(self.df.columns)})'

    @property
    def simulated_df(self):
        assert len(self.df.columns)==2, 'You passed more than 2 series to the model'
        assert any(['TIBR'==i for i in self.df.columns]), 'The DataFrame does not contain TIBR series'

        # Make the Sigmoid transformation:
        if self.trans_par:
            pass 
