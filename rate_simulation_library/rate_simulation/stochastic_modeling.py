#--------------------- Stochastic Modeling Module-----------------------
'''
This module has classes and functions related to the stochastic modeling
of stochastic processes. In particular, the main classes are related to
the Geometric Brownian Motion Monte Carlo simulation.
'''

# ------------------------1. Libraries----------------------------------
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from dateutil.relativedelta import relativedelta
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
            sigma (float): standard deviation or volatility of the GBM.
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
        first_term = (self.mu-(self.sigma**2)/2)*self.dt
        simulated_paths = self.s*np.exp(
            (first_term+self.sigma*brownian_motion)).cumprod(axis=0)
        
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
            upper_bound: float=None, lower_bound: float=None,
            prev_month_sim_path: str=None, hist_begin_year='2011',
            forecast_end_year='2025'
            ):
        """
        Args:
            series (Union[pd.DataFrame, pd.Series]): contains the data 
                of the series that will be modeled.
            Np (int, optional): number of paths simulated. Defaults to 
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
            prev_month_sim_path (string, optional): path to the results 
                of the previous simulation results. This is used to plot
                the comparison between the current month results and the
                previous month results. The file pointed by the path
                must be CSV.
            hist_begin_year (str): beginning year of the historical data
                to be plotted. Defaults to '2011'.
            forecast_end_year (str): final year of the forecasted data
                to be plotted. Defaults to '2025'.

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
                'hist_min': '#08AAB7',
                'hist_max': '#08AAB7'
                }

        # Define attributes:
        self._series_type = 'SeriesRateSimulation'
        self.series = series
        self.COLORS = color_dict
        self.Np= Np
        self.Nt = Nt
        self.T = T
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.prev_month_sim_path = prev_month_sim_path
        self.hist_begin_year = hist_begin_year
        self.forecast_end_year = forecast_end_year
        simulated_df = self.simulate_df()
        vals = np.where(
            simulated_df.values < 0, 
            0,
            simulated_df.values
        )
        self.simulated_df = pd.DataFrame(
            index = simulated_df.index,
            columns = simulated_df.columns,
            data = vals
        )


    
    def __str__(self):
        name = f'{self.series.name}: Np = {self.Np} | Nt = {self.Nt}'
        return '('+self._series_type+') '+ name
    
    def __repr__(self):
        return self.__str__()
    
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
        full_df = pd.concat([observed_df.iloc[:-1, :], self.simulated_df])

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
        variation_series = self.series.diff(periods=delta_t)
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
        var_forecast = self.full_df - self.full_df.shift(delta_t)
        init_date = self.series.index.max()
        var_forecast = var_forecast.loc[init_date:]

        return var_forecast
    
    def plot_hist_forecast(self, 
                           **layout_dict) -> plotly.graph_objs.Figure:
        """Plots the historical behavior of the analized series plus its
        Monte Carlo simulation forecast.

        Returns:
            plotly.graph_objs._figure.Figure: figure with the plot
                information.
        """
        t0_date = self.series.iloc[:-1].index.max()
        mc_agg_df_t0 = pd.read_csv(
            self.prev_month_sim_path,
            parse_dates = [0],
            index_col = 0
            ).loc[t0_date:, :]
        if mc_agg_df_t0.values.max() < 1:
            mc_agg_df_t0 = mc_agg_df_t0 * 100
        mc_agg_df_t0 = self.aggregate_simulations(mc_agg_df_t0)
        mc_agg_df_t1 = self.aggregate_simulations(self.simulated_df)
        fig = plotly_plot_comp(
            hist_series = self.series[self.hist_begin_year:],
            mc_agg_df_t0 = mc_agg_df_t0.loc[:self.forecast_end_year, :],
            mc_agg_df_t1 = mc_agg_df_t1.loc[:self.forecast_end_year],
            color_dict = self.COLORS,
            **layout_dict
        )
        return fig
    
    def plot_variations(self, delta_t: int =1,
                        **layout_dict) -> plotly.graph_objs.Figure:
        """Plots the historical behavior of the analyzed series variations plus
        its Monte Carlo simulation forecast variations.

        Args:
            delta_t (int, Optional): time delta of the variation. 
                Defaults to 1.

        Returns:
            plotly.graph_objs._figure.Figure: figure with the plot
                information.
        """
        var_series = self.variation_series(delta_t=delta_t)
        var_mc = self.variation_forecast_df(delta_t=delta_t)

        var_mc_agg_df = self.aggregate_simulations(var_mc)
        fig = plotly_plot(
            hist_series = var_series[self.hist_begin_year:],
            mc_agg_df = var_mc_agg_df[:self.forecast_end_year],
            color_dict = self.COLORS,
            **layout_dict
        )
        return fig
    
    def plot_simulations(self, n_sims: int = 8, rows: int = 2, cols: int = 4,
                         **layout_dict) -> plotly.graph_objs.Figure:
        """Plots in subplots different simulation paths.

        Args:
            n_sims (int, optional): number of simulations ploted. 
                Defaults to 8.
            rows (int, optional): number of rows in the subplot. 
                Defaults to 2.
            cols (int, optional): number of columns in the subplot. 
                Defaults to 4.

        Returns:
            plotly.graph_objs._figure.Figure: Plotly figure with the 
                data of the plot.
        """
        sims = list(np.random.randint(0, 999, size=n_sims))
        mc_sims = self.simulated_df.iloc[:, sims]
        fig = plot_simulations(
            series = self.series[self.hist_begin_year:], 
            sim_df = mc_sims.loc[:self.forecast_end_year],
            color_dict = self.COLORS,
            rows = rows,
            cols = cols,
            **layout_dict    
        )
        return fig


# 3. Child GBMRateSeries class:

class GBMRateSeries(RateSeriesSimulation):
    """This class stores a rate series, its statistical information
    and with it generates GBM simulations of the rate series. It also 
    has methods to plot historical behavior and its variations.
    """

    def __init__(
            self, series: Union[pd.DataFrame, pd.Series], Np: int=1000, 
            Nt: int=60, T: int=60, color_dict: dict=None, 
            upper_bound: float=None, lower_bound: float=None,
            prev_month_sim_path: str=None, hist_begin_year='2011',
            forecast_end_year='2025'
        ):
        """
        Args:
            series (Union[pd.DataFrame, pd.Series]): contains the data 
                of the series that will be modeled.
            Np (int, optional): number of paths simulated. Defaults to 
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
            prev_month_sim_path (string, optional): path to the results 
                of the previous simulation results. This is used to plot
                the comparison between the current month results and the
                previous month results. The file pointed by the path
                must be CSV.
            hist_begin_year (str): beginning year of the historical data
                to be plotted. Defaults to '2011'.
            forecast_end_year (str): final year of the forecasted data
                to be plotted. Defaults to '2025'.

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
            lower_bound = lower_bound,
            prev_month_sim_path = prev_month_sim_path,
            hist_begin_year = hist_begin_year,
            forecast_end_year = forecast_end_year
            )
        self._series_type = 'GBM Process'

        

    def __str__(self):
        base_info = RateSeriesSimulation.__str__()
        sts_info = f'mean = {self.mu:.2f}, std = {self.sigma:.2f}'
        return base_info+'\n'+sts_info

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
        simulated_df.loc[self.series.index[-1], :] = self.last
        simulated_df.sort_index(inplace=True)

        return simulated_df

class PoissonRateSeries(RateSeriesSimulation):
    """This object stores a rate series (expected as the policy rate of
    a central bank), its statical information, and with it, generates
    a simulation of the rate with a combination of an ARIMA and a 
    Poisson process jump model, to simulate the process by which the
    rate jumps.
    """
    def __init__(
            self, series: pd.Series, Np: int=1000, Nt: int=60, T: int=60, 
            color_dict: dict=None, 
            ref_date: Union[str, datetime]='2008-01-01',
            p: int=4, q: int=4, i: int=0, prev_month_sim_path: str=None,
            hist_begin_year='2011', forecast_end_year='2025'):
        """
        Args:
            series (pd.Series): contains the data of the series that
                will be modeled.
            Np (int, optional): number of paths simulated. Defaults to 
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
            ref_date (Union[str, datetime], optional): initial
                date of analysis to compute the parameter of the model.
                Defaults to '2008-01-01'.
            p (int, optional): number of AR lags for the ARIMA model.
                Defaults to 4.
            q (int, optional): number of MA lags for ARIMA model. 
                Defaults to 4.
            i (int, optional): order of integration in the series for
                the ARIMA model. Defaults to 0.
            prev_month_sim_path (string, optional): path to the results 
                of the previous simulation results. This is used to plot
                the comparison between the current month results and the
                previous month results. The file pointed by the path
                must be CSV.
            hist_begin_year (str): beginning year of the historical data
                to be plotted. Defaults to '2011'.
            forecast_end_year (str): final year of the forecasted data
                to be plotted. Defaults to '2025'
        """
        
        self.p = p
        self.q = q
        self.i = i
        self.ref_date = ref_date
        self.jump_series = series.loc[series/series.shift(1)-1 != 0]
        self.jump_series = self.jump_series[ref_date:]
        self.series = series[ref_date:]

        RateSeriesSimulation.__init__(
            self, self.series, Np, Nt, T, color_dict,
            prev_month_sim_path=prev_month_sim_path, 
            hist_begin_year=hist_begin_year, 
            forecast_end_year=forecast_end_year
        )

        self.series = self.series * 100
        arima_rep = f'ARIMA({self.p}, {self.i}, {self.q})'
        poisson_rep = f'Poisson(lambda = {self.lmbda})' 
        self._series_type = arima_rep+' || '+poisson_rep

    @property
    def lmbda(self):
        """Computes the lambda value for the Poisson process.
        """
        temp_series = self.jump_series.to_frame()
        temp_series['Fecha'] = temp_series.index
        temp_series.reset_index(inplace=True, drop=True)
        cur_date = temp_series['Fecha']
        shift_cd = cur_date.shift(1)
        temp_series['delta_t'] = (
            12 * (cur_date.dt.year - shift_cd.dt.year) +
            cur_date.dt.month - shift_cd.dt.month
        )
        lmbda = 1/temp_series['delta_t'].mean()

        return lmbda

    def poisson_jumps(self, mc_arima_df: pd.DataFrame) -> pd.DataFrame:
        """Takes a Monte Carlo matrix simulation from an ARIMA model,
        and adds Poisson jumps to it, according to the lmbda parameter
        of the modeled time series.

        Args:
            mc_arima_df (pd.DataFrame): Monte Carlo simulations of an
                ARIMA model. 

        Returns:
            pd.DataFrame: Monte Carlo simulations of an ARIMA model with
                Poisson jumps.
        """
        # Prepare ARIMA results:
        size = mc_arima_df.shape
        mc_arima_df = np.concatenate(
            (np.full((1, self.Np), np.NaN), mc_arima_df)
        )

        # Poisson jump stochastic process:
        rnd = np.log(np.random.uniform(size=size))
        jump_sim = 1 + (-rnd/self.lmbda).round(0).cumsum(axis=0)
        jump_sim = np.where(jump_sim > 61, np.NaN, jump_sim)
        jump_sim = pd.DataFrame(data=jump_sim).fillna(method='ffill')
        jump_sim = jump_sim.fillna(0)
        jump_sim = jump_sim.astype(int)

        # Include Poisson jumps into Monte Carlo ARIMA:
        output = np.zeros(size)
        for i in range(2, 62):
            jump_index = (jump_sim.loc[:, :]==i).idxmax().values
            output[i-2, :] = mc_arima_df[jump_index, np.arange(size[1])]
        output = pd.DataFrame(data=output).fillna(method='ffill')
        last_obs = self.series[-1]
        output = output.fillna(last_obs)

        return output

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
        #np.random.seed(1789)#-------------------------------------------------------------
        # Generate the date index of the simulation:
        sim_date_index = pd.date_range(
            start = self.series.index[-1]+relativedelta(months=1),
            end = self.series.index[-1]+relativedelta(months=self.Nt),
            freq = 'M'
            )
        # Generate an inverse sigmoid transformation of the data:
        trans_series = -np.log(0.15/self.jump_series-1)
        change_std = (trans_series-trans_series.shift(1)).std()

        # Generate the ARIMA simulation for the rate:
        model = ARIMA(trans_series.values, order=(self.p, self.i, self.q))
        fitted = model.fit()
        self.arima_summary = fitted.summary()
        trans_forecast = fitted.forecast(60).reshape(60,1)

        # Generate different paths from the ARIMA model:
        mc_trans_forecast = trans_forecast + np.random.normal(
            scale = change_std,
            size = (60, self.Np)
            ).cumsum(axis=0)

        # Bring back to % units the forecast:
        mc_forecast = 0.15/(1+np.exp(-mc_trans_forecast))
        
        # Include Poisson Jumps:
        mc_forecast = self.poisson_jumps(mc_forecast)
        mc_forecast.index = sim_date_index

        # Round to closer 0.25 multiple, with two decimals:
        mc_forecast = np.round(mc_forecast*400, 0)/4
        mc_forecast.loc[self.series.index[-1], :] = self.series[-1]*100
        mc_forecast.sort_index(inplace=True)
        
        return mc_forecast
    
        
class VARModelSeries(GBMRateSeries):
    """This  object stores a rate series, its complementary predictive
    series, its statistical information. It also contains methods and
    attributes that help model is as a bi-varible VAR model.
    """

    def __init__(
            self, df: pd.DataFrame,
            predictive_forecast_df: pd.DataFrame,
            target_name: str,
            predictive_name: str = 'TIBR',
            Np: int = 1000,
            Nt: int = 60, 
            T: int = 60,
            color_dict: dict = None,
            lags: Union[tuple, list] = (1, 2), 
            transformation_parameters: Union[list, tuple] = (0.15, 1),
            prev_month_sim_path: str=None,
            hist_begin_year='2011',
            forecast_end_year='2025'
            ):
        """
        Args:
            df (pd.DataFrame): contains as columns the historical values
                of the target series and its complementary predictive
                series.
            predictive_forecast_df (pd.DataFrame): contains the
                forecasted values of the predictive rate series, which is
                the basis, with the estimated VAR parameters, to
                forecast 
            target_name (str): name of the target variable of the VAR 
                model.
            predictive_name (str, optional): name of the predictive
                variable. Defaults to 'TIBR'.
            Np (int, optional): number of paths simulated. Defaults to 
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
            lags (Union[tuple, list], optional): array-like object with
                the lags of both variables. It is supposed that the
                first set of lags is for the target variable and the
                second for the predictive variable. 
                Defaults to ((1,1),(2,2)).
            transformation_parameters (Union[list, tuple], optional):
                values in the nominator of the sigmoid transformation of 
                the rates. It is expected to have the parameters of 
                transformation ordered as (predictive, target). Defaults
                to (0.15, 1).
            prev_month_sim_path (string, optional): path to the results 
                of the previous simulation results. This is used to plot
                the comparison between the current month results and the
                previous month results. The file pointed by the path
                must be CSV.
            hist_begin_year (str): beginning year of the historical data
                to be plotted. Defaults to '2011'.
            forecast_end_year (str): final year of the forecasted data
                to be plotted. Defaults to '2025'.
        """
        names = [predictive_name, target_name]
        self.lags = lags
        df = df.dropna()
        self.pred_forecast_df = predictive_forecast_df/100
        self.target_name = target_name
        self.predictive_name = predictive_name
        self.trans_parameters = np.array(transformation_parameters)
        self.lag_names = [f'L{i}.{name}' for i in lags for name in names]
        RateSeriesSimulation.__init__(
            self,
            series = df[names],
            Np = Np,
            Nt = Nt,
            T = T,
            color_dict = color_dict,
            prev_month_sim_path = prev_month_sim_path,
            hist_begin_year =  hist_begin_year,
            forecast_end_year = forecast_end_year
        )
        self.series = df[target_name] * 100
        self.pred_series = df[predictive_name] * 100
    
    def __str__(self):
        return f'VAR Moder ({list(self.series.columns)})'

    def simulate_df(self):
        # Make the Sigmoid transformation:
        transformed_series = -np.log(self.trans_parameters/self.series-1)
        transformed_series = transformed_series.dropna().loc['2008':]
        trans_pred_forc_df = -np.log(
            self.trans_parameters[0]/self.pred_forecast_df-1
        )
  
        # Define and train the model:
        model = VAR(transformed_series)
        model_fit = model.fit(maxlags=max(self.lags))
        self.model = model_fit 
        self.var_summary = model_fit.summary()
        self.params = model_fit.params[self.target_name]
        std_error = model_fit.resid[self.target_name].std()

        # Generate the Monte Carlo simulations:
        last_date = transformed_series.index[-1]
        mc_forecast = pd.DataFrame(
            data = np.tile(transformed_series[self.target_name],(self.Np,1)).T,
            index = transformed_series.index
        )
        

        for _ in range(self.Nt):
            last_date = last_day_month(last_date + relativedelta(months=1))
            lag_dates = [
                last_day_month(last_date - relativedelta(months=lag)) for lag 
                in self.lags
            ]
            mc_forecast.loc[last_date, :] = (
                self.var_computation(
                    lag_dates = lag_dates, 
                    lag_names = self.lag_names,
                    mc_df = trans_pred_forc_df,
                    target_df = mc_forecast,
                    params = self.params
                ) + np.random.normal(scale=std_error, size=(1, 1000))
            )
        mc_forecast = mc_forecast.loc[self.series.index[-1]:, :]
        mc_forecast = self.trans_parameters[1]/(1+np.exp(-mc_forecast))

        return mc_forecast*100
    
    @staticmethod
    def var_computation(
            lag_dates: list, lag_names: list, mc_df: pd.DataFrame,
            target_df: pd.DataFrame, params: pd.Series
            ) -> float:
        """This function takes a VAR model parameters, the values of the
        lag dates, a list of lag names, a Monte Carlo simulation
        dataframe for the predictive variable, the target variable
        series to predict the next Monte Carlo values for the target
        variable.

        Args:
            lag_dates (list): list of the lag dates from which the
                VAR equation will be computed.
            lag_names (list): names of the lags that will be searched in
                the params series.
            mc_df (pd.DataFrame): Monte Carlo simulations of the
                predictive variable.
            target_df (pd.DataFrame): target variables historical data.
            params (pd.Series): VAR parameters for the target series.

        Returns:
            float: output of the VAR equation.
        """
        output = params['const']
        i = 0
        for date in lag_dates:
            lag_name = lag_names[i]
            output += mc_df.loc[date, :].values * params[lag_name]
            i += 1
            lag_name = lag_names[i]
            output += target_df.loc[date, :].values * params[lag_name]
            i += 1
        
        return output
    
    def plot_simulations(self, n_sims: int = 8, rows: int = 2, cols: int = 4,
                         **layout_dict) -> plotly.graph_objs.Figure:
        """Plots in subplots different simulation paths.

        Args:
            n_sims (int, optional): number of simulations ploted. 
                Defaults to 8.
            rows (int, optional): number of rows in the subplot. 
                Defaults to 2.
            cols (int, optional): number of columns in the subplot. 
                Defaults to 4.

        Returns:
            plotly.graph_objs._figure.Figure: Plotly figure with the 
                data of the plot.
        """
        last_date = self.series.index[-1]
        sims = list(np.random.randint(0, 999, size=n_sims))
        mc_sims = self.simulated_df.iloc[:, sims]
        pred_mc_sims = (self.pred_forecast_df * 100).iloc[:, sims]
        pred_mc_sims = pred_mc_sims.loc[last_date:, :]

        fig = plot_simulations_with_pred_var(
            series = self.series['2011':],
            pred_series = self.pred_series['2011':], 
            sim_df = mc_sims[:'2025'],
            pred_sim_df = pred_mc_sims[:'2025'],
            color_dict = self.COLORS,
            rows = rows,
            cols = cols,
            **layout_dict    
        )
        return fig

class RandomVariationSeriesSimulation(RateSeriesSimulation):
    """This class simulates a interest rate series that variates in the
    same line as a reference rate. It also adds a random variation. The
    series is modeled by using the variations of the reference rate plus
    some randomness.
    """

    def __init__(
            self, series: Union[pd.DataFrame, pd.Series], 
            mc_reference_df: pd.DataFrame, 
            reference_rate_series: pd.Series, Np: int = 1000, Nt: int = 60, 
            T: int = 60, color_dict: dict = None, upper_bound: float = None, 
            lower_bound: float = None, variation_type: str = 'sum',
            prev_month_sim_path: str=None, hist_begin_year='2011',
            forecast_end_year='2025'
            ):
        """
        Args:
            series (Union[pd.DataFrame, pd.Series]): contains the data
                of the series that will be modeled.
            mc_reference_df (pd.DataFrame): contains the historical data
                and Monte Carlo simulations of the reference rate. From
                this data frame the object computes the monthly
                variations that are applied to the reference rate.
            Np (int, optional): number of paths simulated. Defaults to 
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
            variation_type (str, 'multiplication', 'sum'): defines how
                the variations are computed. 
                
                - 'multiplication': the percentage change is computed
                  and then is multiplied with the last value of the 
                  modeled rate.

                - 'sum': the difference is computed and then added to 
                  the modeled rate.

                Defaults to 'sum'.
            prev_month_sim_path (string, optional): path to the results 
                of the previous simulation results. This is used to plot
                the comparison between the current month results and the
                previous month results. The file pointed by the path
                must be CSV.
            hist_begin_year (str): beginning year of the historical data
                to be plotted. Defaults to '2011'.
            forecast_end_year (str): final year of the forecasted data
                to be plotted. Defaults to '2025'.
        """
        variation_type = variation_type.lower()
        var_types = ('multiplication', 'sum')
        assert variation_type in var_types, f'Invalid var_type passed. Valid types are {var_types}'

        series = series.dropna()
        last_date = series.index[0]
        self.reference_rate_series = reference_rate_series[last_date:] * 100
        self.mc_reference_df = mc_reference_df/100
        self.variation_type = variation_type

        super().__init__(
            series = series,
            Np = Np,
            Nt = Nt,
            T = T,
            color_dict = color_dict,
            upper_bound = upper_bound,
            lower_bound = lower_bound,
            prev_month_sim_path = prev_month_sim_path,
            hist_begin_year =  hist_begin_year,
            forecast_end_year = forecast_end_year
        )
        self.series = self.series*100
        
    def simulate_df(self) -> pd.DataFrame:
        """With the mc_reference_df attribute, computes the 1 period
        variation and then applies that variation to the modeled rate.
        How this is done is determined by the 'variation_type'
        attribute.

        Returns:
            pd.DataFrame: contains the simulated paths of the models
                rate.
        """
        last_observed_date = self.series.index[-1]
        last_target_val = self.series[-1]
        mc_reference = self.mc_reference_df.loc[last_observed_date:, :]

        if self.variation_type == 'sum':
            var_df = (mc_reference - mc_reference.shift(1)).dropna()
            rnd_val = np.random.normal(0, 0.001, size=var_df.shape)
            simulated_df = (
                last_target_val + 
                (var_df + rnd_val).cumsum()
            )
        else:
            var_df = (mc_reference/mc_reference.shift(1)).dropna()
            rnd_val = np.random.normal(0, 0.001, size=var_df.shape)
            simulated_df = (
                last_target_val * 
                (var_df + rnd_val).cumprod()
            )
        
        simulated_df.loc[last_observed_date, :] = last_target_val
        simulated_df.sort_index(inplace=True)
        return simulated_df * 100
    
    def plot_simulations(self, n_sims: int = 8, rows: int = 2, cols: int = 4,
                         **layout_dict) -> plotly.graph_objs.Figure:
        """Plots in subplots different simulation paths.

        Args:
            n_sims (int, optional): number of simulations plotted. 
                Defaults to 8.
            rows (int, optional): number of rows in the subplot. 
                Defaults to 2.
            cols (int, optional): number of columns in the subplot. 
                Defaults to 4.

        Returns:
            plotly.graph_objs._figure.Figure: Plotly figure with the 
                data of the plot.
        """
        sims = list(np.random.randint(0, 999, size=n_sims))
        mc_sims = self.simulated_df.iloc[:, sims]
        pred_mc_sims = (self.mc_reference_df * 100).iloc[:, sims]
        fig = plot_simulations_with_pred_var(
            series = self.series['2011':],
            pred_series = self.reference_rate_series['2011':], 
            sim_df = mc_sims[:'2025'],
            pred_sim_df = pred_mc_sims[:'2025'],
            color_dict = self.COLORS,
            rows = rows,
            cols = cols,
            **layout_dict    
        )
        return fig
    

class UVRStochasticModel(RateSeriesSimulation):
    """This class models the stochastic behavior of the UVR based on the
    monthly change of the inflation index.
    """
    
    def __init__(
            self, uvr_series: pd.Series, index_series: pd.Series,
            mc_yearly_ipc: pd.DataFrame, ipc_series: pd.Series, 
            color_dict: dict = None, upper_bound: float = None, 
            lower_bound: float = None, prev_month_sim_path: str=None, 
            hist_begin_year='2011',forecast_end_year='2025'
            ):
        """
        Args:
            uvr_series (pd.Series): contains the daily series of UVR.
            index_series (pd.Series): contains the monthly series of the
                IPC index.
            mc_yearly_ipc (pd.DataFrame): contains the Np Monte Carlo
                simulations of the yearly IPC change rate.
            ipc_series (pd.Series): contains the historical monthly
                behavior of the IPC yearly change.
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
            prev_month_sim_path (string, optional): path to the results 
                of the previous simulation results. This is used to plot
                the comparison between the current month results and the
                previous month results. The file pointed by the path
                must be CSV.
            hist_begin_year (str): beginning year of the historical data
                to be plotted. Defaults to '2011'.
            forecast_end_year (str): final year of the forecasted data
                to be plotted. Defaults to '2025'.
        """
        uvr_series = uvr_series.dropna()
        monthly_uvr = uvr_series.groupby(pd.Grouper(freq='M')).last()
        self.uvr_2w_series = uvr_series[
            (uvr_series.index.day == 15) |
            (uvr_series.index.isin(list(monthly_uvr.index)))
        ]
        self.index_series = index_series.dropna()
        self.ipc_series = ipc_series.dropna()
        self.mc_yearly_ipc = mc_yearly_ipc.dropna()

        super().__init__(
            series = monthly_uvr,
            color_dict = color_dict,
            upper_bound = upper_bound,
            lower_bound = lower_bound,
            prev_month_sim_path = prev_month_sim_path,
            hist_begin_year =  hist_begin_year,
            forecast_end_year = forecast_end_year
        )

    def simulate_df(self) -> pd.DataFrame:
        """Simulates the behavior of the UVR, using the Monte Carlo
        simulations of the yearly IPC applied to the IPC index series to
        obtain the monthly inflation rate. With the monthly inflation
        rate, and using the UVR formula defined by the BanRep, compute
        the Monte Carlo behavior of the UVR.

        Returns:
            pd.DataFrame: Monte Carlo simulations of the UVR rate.
        """
        # Compute the monthly inflation rate:
        index_df = pd.DataFrame(
            data = np.tile(self.index_series, (1000, 1)).T,
            index = self.index_series.index
        )
        for date in self.mc_yearly_ipc.index[:-1]:
            index_df.loc[date, :] = 0
            vals = (
                (self.mc_yearly_ipc.loc[date, :].values/100 + 1) *
                index_df.shift(12).loc[date, :].values
            )
            index_df.loc[date, :] = vals
        self.index_df = index_df
        monthly_ipc = index_df.pct_change().dropna()
        monthly_ipc = monthly_ipc.loc[self.index_series.index[-2]:, :].round(4)
        self.monthly_ipc = monthly_ipc * 100

        # Compute the Monte Carlo paths of the UVR:
        monthly_uvr_df = pd.DataFrame(
            data = np.tile(self.series, (1000, 1)).T,
            index = self.series.index
        )

        # Compute first forecast 15th day UVR:
        date = self.series.index[-1]
        ref_date = last_day_month(date - relativedelta(months=1))
        end_date = date + relativedelta(days=15)
        start_date = end_date - relativedelta(months=1)
        start_uvr = self.uvr_2w_series[start_date]
        uvr_15 = (
            start_uvr * (1 + monthly_ipc.loc[ref_date, :].values)
            ).round(4)

        for date in monthly_ipc.index[2:]:
            end_date = date + relativedelta(days=15)
            start_date = end_date - relativedelta(months=1)
            ref_date = last_day_month(date - relativedelta(months=1))

            # Last day UVR:
            t = (date - start_date).days
            d = (end_date - start_date).days
            e = t/d
            ipc = monthly_ipc.loc[ref_date, :].values
            last_uvr = uvr_15 * (1 + ipc) ** e

            monthly_uvr_df.loc[date, :] = last_uvr.round(4)

            # 15th day UVR:
            uvr_15 = (uvr_15 * (1 + ipc)).round(4)

        return monthly_uvr_df.loc[self.series.index[-1]:, :]

    def plot_simulations(self, n_sims: int = 8, rows: int = 2, cols: int = 4,
                         **layout_dict) -> plotly.graph_objs.Figure:
        """Plots in subplots different simulation paths.

        Args:
            n_sims (int, optional): number of simulations plotted. 
                Defaults to 8.
            rows (int, optional): number of rows in the subplot. 
                Defaults to 2.
            cols (int, optional): number of columns in the subplot. 
                Defaults to 4.

        Returns:
            plotly.graph_objs._figure.Figure: Plotly figure with the 
                data of the plot.
        """
        sims = list(np.random.randint(0, 999, size=n_sims))
        uvr_12m_var = self.full_df.pct_change(periods=12)
        uvr_12m_var = uvr_12m_var.loc[self.series.index[-1]:, sims] * 100
        hist_12m_var = self.series.pct_change(periods=12).dropna() * 100
        pred_mc_sims = self.mc_yearly_ipc.loc[:, sims]
        fig = plot_simulations_with_pred_var(
            series = hist_12m_var['2011':],
            pred_series = self.ipc_series['2011':], 
            sim_df = uvr_12m_var[:'2025'],
            pred_sim_df = pred_mc_sims[:'2025'],
            color_dict = self.COLORS,
            rows = rows,
            cols = cols,
            **layout_dict    
        )
        return fig

class RNNSeriesSimulation(RateSeriesSimulation):
    """This class uses a recurrent neural network to forecast a rate
    series, stores its information. It also has methods to plot
    historical behavior and its variations.
    """

    def __init__(
        self, data: pd.DataFrame, checkpoint_path: str, 
        target_series: pd.Series, Np: int=1000, Nt: int=60, T: int=60, 
        color_dict: dict=None):

        self.checkpoint_path = checkpoint_path
        # Prepare series 
        self.series = target_series
        self.data = data
        self.diff_series = data.merge(target_series, left_index=True,
                                      right_index=True)
        self.diff_series = self.diff_series['FFR'] - self.diff_series['SOFR']

        super().__init(self, self.series, Np, Nt, T, color_dict)

    
    def simulate_df(self) -> pd.DataFrame:
        end_date = self.series.index[-1] + relativedelta(years=5)
        str_end_date = end_date.strftime('%Y-%m-%d')
        model, checkpoint = load_checkpoint(self.checkpoint_path)
        ffr_forecast = generate_rnn_forecast(model, self.data, str_end_date)
        randomness = np.random.normal(0, self.diff_series.std(), 
                                      size=(61, 1000))
        forecast = ffr_forecast.loc[self.series.index[-1]:, :] + randomness
        return forecast * 100

