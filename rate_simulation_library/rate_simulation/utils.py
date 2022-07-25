#---------------------------- Utils Module ------------------------------------
"""
This utils module is a complementary module for the library, which
contains additional functions needed for the correct functioning of
the library.
"""
# ------------------------ Import Libraries -----------------------------------
from matplotlib.pyplot import hist
import numpy as np
import pandas as pd
from typing import Union
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
from datetime import datetime

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch

#-------------------------- Global Variables ----------------------------------
FRED_KEY = "7b8d5df532fc2c51afd6579e63e52cdd"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------- Classes ---------------------------------------
class RecurrentNet(nn.Module):
    def __init__(
            self, input_dim: int, 
            output_dim: int, 
            recurrent_layer: str,
            num_rnn_layers: int,
            rnn_hidden_units: int,
            dropout_bool: bool = 0,
            dropout: float = False,
            bn0_bool: bool = True,
            bn1_bool: bool = True) -> nn.Module:
        # Init parent class:
        super().__init__()
        layers = []

        # Define hyper-parameters:
        self.bn0_bool = bn0_bool
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_units = rnn_hidden_units
        self.recurrent_layer = recurrent_layer

        # Define layers:
        self.bn0 = nn.BatchNorm1d(input_dim)
        rnn_layer = getattr(nn, recurrent_layer)
        if dropout_bool:
            self.rnn = rnn_layer(
                input_size = input_dim,
                hidden_size = rnn_hidden_units,
                num_layers = num_rnn_layers,
                batch_first = True,
                dropout = dropout
            )
        else:
            self.rnn = rnn_layer(
                input_size = input_dim,
                hidden_size = rnn_hidden_units,
                num_layers = num_rnn_layers,
                batch_first = True
            )
        if bn1_bool:
            batch_norm = nn.BatchNorm1d(rnn_hidden_units)
            layers.append(batch_norm)
        dense = nn.Linear(rnn_hidden_units, output_dim)
        layers.append(dense)
        self.dense = nn.Sequential(*layers)

    def forward(self, sequence):
        batch_size = sequence.shape[0]
        if self.bn0_bool:
            sequence = sequence.permute(0, 2, 1)
            sequence = self.bn0(sequence)
            sequence = sequence.permute(0, 2, 1)
            
        if self.recurrent_layer == 'LSTM':
            (stm, ltm) = self.init_hidden(batch_size)
            _, hidden = self.rnn(sequence, (stm.detach(), ltm.detach()))
            stm, ltm = hidden
            output = torch.sigmoid(self.dense(stm[-1, :, :]))
        
        else:
            h_c = self.init_hidden(batch_size)
            _, h_c = self.rnn(sequence, h_c)
            output = torch.sigmoid(self.dense(h_c[-1, :, :]))

        return output
    
    def init_hidden(self, batch_size):
        """Initializes hidden state of the LSTM layers"""
        weight = next(self.parameters()).data
        weight = weight.new(self.num_rnn_layers, batch_size, 
                            self.rnn_hidden_units)
        gain = nn.init.calculate_gain('tanh')
        stm = torch.nn.init.xavier_uniform_(weight, gain=gain).to(DEVICE)
        
        if self.recurrent_layer =='GRU':
            return stm
        else:
            ltm = torch.nn.init.xavier_uniform_(weight, gain=gain).to(DEVICE)
            hidden = (stm, ltm)
            return hidden

#----------------------- Complementary Functions ------------------------------
# 1. Grouping functions:
def quant_5(x: np.array)->float:
    """From an array returns the 5th percentile

    Args:
        x (np.array): array of floats

    Returns:
        float: 5th percentile value from the array of floats
    """
    return x.quantile(0.05)

def quant_95(x: np.array)->float:
    """From an array returns the 95th percentile

    Args:
        x (np.array): array of floats

    Returns:
        float: 95th percentile value from the array of floats
    """
    return x.quantile(0.95)


# 2. Miscellaneous functions:

def prop_label(x:Union[float, int])->float:
    """Takes a specific proportion of the input value. This function is
    used in particular to possition the label in the plots of the
    different model classes.

    Args:
        x (Union[float, int]): numerical value

    Returns:
        float: proportion of the input x
    """
    if x<0: return 1.3*x
    else: return 1.05*x

def jump_class(x: Union[float, int])->int:
    """Classifies the type of jump for the Poisson model, so that if the
    condition is met, it goes to the determined below class.

    Args:
        x (Union[float, int]): number of months between jumps

    Returns:
        int: classification number for the lmbda computation.
    """
    if x >= 5: return 5
    elif x == 0: return 1
    else: return x

def last_day_month(date):
    nd = date.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
    return nd

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = RecurrentNet(
        input_dim = 2,
        output_dim = 2,
        recurrent_layer = checkpoint['recurrent_type'],
        num_rnn_layers = checkpoint['num_rnn_layers'],
        rnn_hidden_units = checkpoint['rnn_hidden_units']
    )
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint
    
# 3. Plot functions:

def plotly_plot(
        hist_series: Union[pd.DataFrame, pd.Series], 
        mc_agg_df: pd.DataFrame,
        color_dict: dict = None,
        **layout_dict,
    ) -> plotly.graph_objs.Figure:
    """This functions creates a Plotly Figure object with the historical
    data of the input series, combined with the Monte Carlo simulations
    of its future behavior. It also has the historical maximum value and
    the historical minimum value of the series added to the plot.

    Args:
        hist_series (Union[pd.DataFrame, pd.Series]): historical
            values of the plotted series.
        mc_agg_df (pd.DataFrame): Monte Carlo simulations of the series,
            which is summarized by the mean, maximum, minimum, 5th
            percentile and 95th percentile for each time step. Its
            column names should be:

            - 'max': for the column with the maximum value per time step
            - 'min': for the column with the minimum value per time step
            - 'mean': for the column with the mean value per time step
            - 'quant_5': for the column with the 5th percentile value 
              per time step.
            - 'quant_95': for the column with the 95th percentile value 
              per time step.
        color_dict (dict): dictionary with the colors for the plotted 
            lines.

    Returns:
        plotly.graph_objs._figure.Figure: figure with the plot
            information.
    """
    # Find historical values:
    hist_min = hist_series.min()
    hist_min_date = hist_series.idxmin().strftime('%b %Y')
    hist_max = hist_series.max()
    hist_max_date = hist_series.idxmax().strftime('%b %Y')

    # Create Figure object:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = mc_agg_df.index,
            y = mc_agg_df['max'],
            mode = 'lines',
            name = 'Max',
            line = {'color': color_dict['max'], 'width': 2, 'dash': 'dot'},
            showlegend = True
        )
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df.index,
            y = mc_agg_df['quant_95'],
            mode = 'lines',
            name = 'Pct.95',
            line = {'color': color_dict['perc_95'], 'width': 2, 'dash': 'dot'}
        )
    )
    fig.add_trace(
        go.Scatter(
            x = mc_agg_df.index,
            y = mc_agg_df['mean'],
            mode = 'lines',
            name = 'Media',
            line = {'color': color_dict['mean'], 'width': 2, 'dash': 'dash'}
        )
    )
    fig.add_trace(
        go.Scatter(
            x = hist_series.index,
            y = hist_series,
            mode = 'lines',
            name = 'Hist.',
            line = {'color': color_dict['hist'], 'width': 2},
            showlegend = True
        )
    )


    fig.add_trace(
        go.Scatter(
            x = mc_agg_df.index,
            y = mc_agg_df['quant_5'],
            mode = 'lines',
            name = 'Pct.5',
            line = {'color': color_dict['perc_5'], 'width': 2, 'dash': 'dot'}
        )
    )
    fig.add_trace(
        go.Scatter(
            x = mc_agg_df.index,
            y = mc_agg_df['min'],
            mode = 'lines',
            name = 'Min',
            line = {'color': color_dict['min'], 'width': 2, 'dash': 'dot'}
        )
    )

    fig.update_layout(**layout_dict)
    fig.add_hline(
        y = hist_max,
        line_dash = 'dashdot',
        line_color = color_dict['hist_max'],
        annotation = {
            'text': hist_max_date+' = '+str(round(hist_max, 2)),
            'font': {'color': color_dict['hist_max'], 'size': 13}
        },
        annotation_position = 'top left'
    )
    fig.add_hline(
        y = hist_min,
        line_dash = 'dashdot',
        line_color = color_dict['hist_min'],
        annotation = {
            'text': hist_min_date+' = '+str(round(hist_min, 2)),
            'font': {'color': color_dict['hist_min'], 'size': 13}
        },
        annotation_position = 'bottom left'
    )
    return fig

def plot_simulations(series: pd.Series, color_dict: dict,
                     sim_df: pd.DataFrame, rows: int = 2, cols: int = 4, 
                     **layout_dict) -> plotly.graph_objs.Figure:
    """Plots in subplots different simulation paths.

    Args:
        series (pd.Series): historical data.
        color_dict (dict): dictionary with default colors.
        sim_df (pd.DataFrame): dataframe with simulated paths.
        rows (int, optional): number of rows in the subplot. Defaults to
             2.
        cols (int, optional): number of columns in the subplot. Defaults 
            to 4.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure with the data of
            the plot.
    """
    subplot_titles = [f'Simulación {i+1}' for i in range(sim_df.shape[1])]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for i in range(sim_df.shape[1]):
        row = i//cols+1
        col = i%cols+1
        fig.add_trace(
            go.Scatter(
                x = sim_df.iloc[:, i].index,
                y = sim_df.iloc[:, i],
                mode = 'lines',
                name = 'Media',
                showlegend = False,
                line = {
                    'color': color_dict['mean'], 
                    'width': 2, 
                    'dash': 'dash'
                }
            ),
            row = row,
            col = col
        )
        fig.add_trace(
            go.Scatter(
                x = series.index,
                y = series,
                mode = 'lines',
                name = 'Hist.',
                line = {'color': color_dict['hist'], 'width': 2},
                showlegend = False
            ),
            row = row,
            col = col
        )
    
    fig.update_layout(**layout_dict)

    return fig

def plot_simulations_with_pred_var(
            series: pd.Series, pred_series: pd.Series, 
            color_dict: dict, sim_df: pd.DataFrame, 
            pred_sim_df: pd.DataFrame, rows: int = 2, cols: int = 4, 
            **layout_dict
            ) -> plotly.graph_objs.Figure:
    """Plots in subplots different simulation paths.

    Args:
        series (pd.Series): historical data of target variable.
        pred_series (pd.Series): historical data of predictive variable.
        color_dict (dict): dictionary with default colors.
        sim_df (pd.DataFrame): dataframe with simulated paths of
            target variable.
        pred_sim_df (pd.DataFrame): dataframe with simulated paths
            predictive variable.
        rows (int, optional): number of rows in the subplot. Defaults to
             2.
        cols (int, optional): number of columns in the subplot. Defaults 
            to 4.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure with the data of
            the plot.
    """
    subplot_titles = [f'Simulación {i+1}' for i in range(sim_df.shape[1])]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for i in range(sim_df.shape[1]):
        if i < 1:
            showlegend = True
        else:
            showlegend = False
        row = i//cols+1
        col = i%cols+1
        fig.add_trace(
            go.Scatter(
                x = pred_sim_df.iloc[:, i].index,
                y = pred_sim_df.iloc[:, i],
                mode = 'lines',
                name = pred_series.name,
                showlegend = False,
                line = {
                    'color': '#0D94CF', 
                    'width': 2, 
                    'dash': 'dash'
                }
            ),
            row = row,
            col = col
        )
        fig.add_trace(
            go.Scatter(
                x = pred_series.index,
                y = pred_series,
                mode = 'lines',
                name = pred_series.name,
                line = {'color': '#0D94CF', 'width': 2},
                showlegend = showlegend
            ),
            row = row,
            col = col
        )
        fig.add_trace(
            go.Scatter(
                x = sim_df.iloc[:, i].index,
                y = sim_df.iloc[:, i],
                mode = 'lines',
                name = series.name,
                showlegend = False,
                line = {
                    'color': color_dict['mean'], 
                    'width': 2, 
                    'dash': 'dash'
                }
            ),
            row = row,
            col = col
        )
        fig.add_trace(
            go.Scatter(
                x = series.index,
                y = series,
                mode = 'lines',
                name = series.name,
                line = {'color': color_dict['hist'], 'width': 2},
                showlegend = showlegend
            ),
            row = row,
            col = col
        )
    
    fig.update_layout(**layout_dict)

    return fig

def plotly_plot_comp(
        hist_series: Union[pd.DataFrame, pd.Series], 
        mc_agg_df_t0: pd.DataFrame,
        mc_agg_df_t1: pd.DataFrame,
        color_dict: dict = None,
        **layout_dict,
    ) -> plotly.graph_objs.Figure:
    """This functions creates a Plotly Figure object with the historical
    data of the input series, combined with the Monte Carlo simulations
    of its future behavior. It also has the historical maximum value and
    the historical minimum value of the series added to the plot.

    Args:
        hist_series (Union[pd.DataFrame, pd.Series]): historical
            values of the plotted series.
        mc_agg_df_t0 (pd.DataFrame): Monte Carlo simulations of the 
            series, which is summarized by the mean, maximum, minimum, 
            5th percentile and 95th percentile for each time step. This 
            is for the t0 results. Its column names should be:

            - 'max': for the column with the maximum value per time step
            - 'min': for the column with the minimum value per time step
            - 'mean': for the column with the mean value per time step
            - 'quant_5': for the column with the 5th percentile value 
              per time step.
            - 'quant_95': for the column with the 95th percentile value 
              per time step.
        mc_agg_df_t1 (pd.DataFrame): Monte Carlo simulations of the 
            series, which is summarized by the mean, maximum, minimum, 
            5th percentile and 95th percentile for each time step. This 
            is for the t1 results. Its column names should be:

            - 'max': for the column with the maximum value per time step
            - 'min': for the column with the minimum value per time step
            - 'mean': for the column with the mean value per time step
            - 'quant_5': for the column with the 5th percentile value 
              per time step.
            - 'quant_95': for the column with the 95th percentile value 
              per time step.
        color_dict (dict): dictionary with the colors for the plotted 
            lines.

    Returns:
        plotly.graph_objs._figure.Figure: figure with the plot
            information.
    """
    # Find historical values:
    hist_min_t0 = hist_series.iloc[:-1].min()
    hist_min_date_t0 = hist_series.iloc[:-1].idxmin().strftime('%b %Y')
    hist_max_t0 = hist_series.iloc[:-1].max()
    hist_max_date_t0 = hist_series.iloc[:-1].idxmax().strftime('%b %Y')

    hist_min = hist_series.min()
    hist_min_date = hist_series.idxmin().strftime('%b %Y')
    hist_max = hist_series.max()
    hist_max_date = hist_series.idxmax().strftime('%b %Y')

    max_date_t0 = hist_series.iloc[:-1].index.max()
    max_date_t1 = hist_series.index.max()

    # Create Figure object:
    titles = [max_date_t0.strftime('%b %Y'), 
              max_date_t1.strftime('%b %Y')]
    fig = make_subplots(rows=1, cols=2, subplot_titles=titles)

    # Left plot----------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t0.index,
            y = mc_agg_df_t0['max'],
            mode = 'lines',
            name = 'Max',
            line = {'color': color_dict['max'], 'width': 2, 'dash': 'dot'},
            showlegend = True,
            legendgrouptitle_text = max_date_t0.strftime('%b %Y'),
            legendgroup = 'prev'
        ), 
        row = 1,
        col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t0.index,
            y = mc_agg_df_t0['quant_95'],
            mode = 'lines',
            name = 'Pct.95',
            line = {'color': color_dict['perc_95'], 'width': 2, 'dash': 'dot'},
            legendgrouptitle_text = max_date_t0.strftime('%b %Y'),
            legendgroup = 'prev'
        ),
        row = 1,
        col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t0.index,
            y = mc_agg_df_t0['mean'],
            mode = 'lines',
            name = 'Media',
            line = {'color': color_dict['mean'], 'width': 2, 'dash': 'dash'},
            legendgrouptitle_text = max_date_t0.strftime('%b %Y'),
            legendgroup = 'prev'
        ),
        row = 1,
        col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = hist_series.loc[:max_date_t0].index,
            y = hist_series.loc[:max_date_t0],
            mode = 'lines',
            name = 'Hist.',
            line = {'color': color_dict['hist'], 'width': 2},
            showlegend = True,
            legendgrouptitle_text = max_date_t0.strftime('%b %Y'),
            legendgroup = 'prev'
        ),
        row = 1,
        col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t0.index,
            y = mc_agg_df_t0['quant_5'],
            mode = 'lines',
            name = 'Pct.5',
            line = {'color': color_dict['perc_5'], 'width': 2, 'dash': 'dot'},
            legendgrouptitle_text = max_date_t0.strftime('%b %Y'),
            legendgroup = 'prev'
        ),
        row = 1,
        col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t0.index,
            y = mc_agg_df_t0['min'],
            mode = 'lines',
            name = 'Min',
            line = {'color': color_dict['min'], 'width': 2, 'dash': 'dot'},
            legendgrouptitle_text = max_date_t0.strftime('%b %Y'),
            legendgroup = 'prev'
        ),
        row = 1,
        col = 1
    )

    fig.add_hline(
        y = hist_max_t0,
        line_dash = 'dashdot',
        line_color = color_dict['hist_max'],
        annotation = {
            'text': hist_max_date_t0+' = '+str(round(hist_max_t0, 2)),
            'font': {'color': color_dict['hist_max'], 'size': 13}
        },
        annotation_position = 'top left',
        row = 1,
        col = 1
    )

    fig.add_hline(
        y = hist_min_t0,
        line_dash = 'dashdot',
        line_color = color_dict['hist_min'],
        annotation = {
            'text': hist_min_date_t0+' = '+str(round(hist_min_t0, 2)),
            'font': {'color': color_dict['hist_min'], 'size': 13}
        },
        annotation_position = 'bottom left',
        row = 1,
        col = 1
    )


    # Right plot---------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t1.index,
            y = mc_agg_df_t1['max'],
            mode = 'lines',
            name = 'Max',
            line = {'color': color_dict['max'], 'width': 2, 'dash': 'dot'},
            showlegend = True,
            legendgrouptitle_text = max_date_t1.strftime('%b %Y'),
            legendgroup = 'act'
        ),
        row = 1,
        col = 2
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t1.index,
            y = mc_agg_df_t1['quant_95'],
            mode = 'lines',
            name = 'Pct.95',
            line = {'color': color_dict['perc_95'], 'width': 2, 'dash': 'dot'},
            legendgrouptitle_text = max_date_t1.strftime('%b %Y'),
            legendgroup = 'act'
        ),
        row = 1,
        col = 2
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t1.index,
            y = mc_agg_df_t1['mean'],
            mode = 'lines',
            name = 'Media',
            line = {'color': color_dict['mean'], 'width': 2, 'dash': 'dash'},
            legendgrouptitle_text = max_date_t1.strftime('%b %Y'),
            legendgroup = 'act'
        ),
        row = 1,
        col = 2
    )

    fig.add_trace(
        go.Scatter(
            x = hist_series.index,
            y = hist_series,
            mode = 'lines',
            name = 'Hist.',
            line = {'color': color_dict['hist'], 'width': 2},
            showlegend = True,
            legendgrouptitle_text = max_date_t1.strftime('%b %Y'),
            legendgroup = 'act'
        ),
        row = 1,
        col = 2
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t1.index,
            y = mc_agg_df_t1['quant_5'],
            mode = 'lines',
            name = 'Pct.5',
            line = {'color': color_dict['perc_5'], 'width': 2, 'dash': 'dot'},
            legendgrouptitle_text = max_date_t1.strftime('%b %Y'),
            legendgroup = 'act'
        ),
        row = 1,
        col = 2
    )

    fig.add_trace(
        go.Scatter(
            x = mc_agg_df_t1.index,
            y = mc_agg_df_t1['min'],
            mode = 'lines',
            name = 'Min',
            line = {'color': color_dict['min'], 'width': 2, 'dash': 'dot'},
            legendgrouptitle_text = max_date_t1.strftime('%b %Y'),
            legendgroup = 'act'
        ),
        row = 1,
        col = 2
    )

    fig.add_hline(
        y = hist_max,
        line_dash = 'dashdot',
        line_color = color_dict['hist_max'],
        annotation = {
            'text': hist_max_date+' = '+str(round(hist_max, 2)),
            'font': {'color': color_dict['hist_max'], 'size': 13}
        },
        annotation_position = 'top left',
        row = 1,
        col = 2
    )

    fig.add_hline(
        y = hist_min,
        line_dash = 'dashdot',
        line_color = color_dict['hist_min'],
        annotation = {
            'text': hist_min_date+' = '+str(round(hist_min, 2)),
            'font': {'color': color_dict['hist_min'], 'size': 13}
        },
        annotation_position = 'bottom left',
        row = 1,
        col = 2
    )

    fig.update_layout(**layout_dict)
    return fig

# 4. Time series functions:

def series_to_sequence(
        data: np.ndarray,
        window_size: int = 6, 
        test_split: float = 0.3
    ) -> tuple:
    """This function transforms series of historical or sequential
    data into an ordered sequence of sequences. The output contains a
    sequence of sequences of size equal to window_size. The input data 
    is also divided into train and test datasets so that the last
    train_test_split proportion of data is part of the test dataset.

    Args:
        data (np.ndarray): array-like
            data structure with the series of reshaped data.
        window_size (int, optional): size of the sequence window. 
            Defaults to 6.
        test_split (float, optional): proportion of the tail of the 
            input data that is assigned as the test dataset. Defaults to
            0.3 (30%). 

    Returns:
        tuple: tuple of arrays (x_train, y_train, x_test, y_test)
        
            - x_train: np.ndarray with the predictive variables. Its
                shape is 
                ((data.shape[0] - window_size) * (1 - test_split), 
                # of variables).

            - y_train: np.array with the target variables. Its shape is
                ((data.shape[0] - window_size) * (1 - test_split), 
                # of variables).
            
            - x_test: np.ndarray with the test predictive variables.

            - y_test: np.ndarray with the test target variables.  
    """
    data = data.values
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # Create the sequence of window_size sized sequences:
    data_t = []
    for index in range(len(data) - window_size):
        data_t.append(data[index: index + window_size + 1])
    
    # Convert list of sequences into array:
    data_t = np.array(data_t)

    # Train-test split:
    split = int(data_t.shape[0] * (1 - test_split))
    x_train = data_t[:split, :-1, :]
    y_train = data_t[:split, -1, :]
    x_test = data_t[split:, :-1, :]
    y_test = data_t[split:, -1, :]

    return (x_train, y_train, x_test, y_test)

def series_to_table(
        data: np.ndarray,
        lags: int = 6,
        test_split: float = 0.3
    )-> tuple:
    """Converts an array of serial values to a concatenated array where
    the additional columns are the lags of that values.

    Args:
        data (np.ndarray): array with the series data.
        lags (int, optional): number of lags. Defaults to 6.
        test_split (float, optional): proportion of the test dataset.
             Defaults to 0.3.

    Returns:
        tuple: tuple of arrays with x_train, y_train, x_test, y_test.
    """
    # Define varibles:
    n_vars = data.shape[1]

    # Concatenate series data with lags:
    data = pd.concat([data.shift(i) for i in range(lags+1)], axis=1)
    data = data.dropna().values

    # Train-test split:
    split = int(data.shape[0] * (1 - test_split))
    x_train = data[:split, n_vars:]
    y_train = data[:split, :n_vars]
    x_test = data[split:, n_vars:]
    y_test = data[split:, :n_vars]

    return (x_train, y_train, x_test, y_test)      

# 4. Forecast functions:
def generate_rnn_forecast(
        model: nn.Module,
        hist_data_df: pd.DataFrame,
        final_date: str,
        time_window: int
        ) -> pd.DataFrame:
    """Takes a model, historical daily-data and returns a forecast that
    until the final_date date.

    Args:
        model (nn.Module): PyTorch neural network model, with which
            the forecast is done.
        hist_data_df (pd.DataFrame): contains the historical sequential
            data
        final_date (str): "%Y-%m-%d" format, is the last date of
            forecast
        time_window (int): number of lags used in the model

    Returns:
        pd.DataFrame: contains the historical data and the forecast
            at the end
    """
    df = hist_data_df.copy()
    end_date = datetime.strptime(final_date, '%Y-%m-%d')
    last_date = df.index[-1]
    num_days = (end_date - last_date).days
    
    for day in range(1, num_days+1):
        array = np.expand_dims(df.iloc[-60:].values, axis=0)
        input_tensor = torch.tensor(array, dtype=torch.float, device=DEVICE)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            output = output.to('cpu').numpy()[0]
            step_date = last_date + relativedelta(days=day)
            df.loc[step_date, :] = output
    
    return df

# Plotting unused functions:
    # def plot_full_series(
    #     self, start: Union[str, datetime.datetime], 
    #     end: Union[str, datetime.datetime], figsize: tuple=(15,10),
    #     time_space: float=2., dec: int=1,
    #     upper_space: int=0.1, lower_space: float=0.1
    #     ):
    #     """Plots the historical and forecasted values of the analized 
    #     series, with the mean, min, max, 5th and 95th percentiles of the
    #     simulated steps.

    #     Args:
    #         start (Union[str, datetime.datetime]): strin with the
    #             beginning date of the plot. Expected format: '%Y-%m-%d'.
    #         end (Union[str, datetime.datetime]): strin with the 
    #             beginning date of the plot. Expected format: '%Y-%m-%d'.
    #         figsize (tuple, optional): [description]. Defaults to 
    #             (15,10).
    #         time_space (float, optional): value of time units along the
    #             x-axis the labels will be possitioned. Defaults to 2.
    #         dec (int, optional): number of decimals in the labels of the
    #             plot. Defaults to 1.

    #     Returns:
    #         tuple: tuple with mlp.figure.Figure, 
    #             mlp..axes._subplots.AxesSubplot which contain the plot
    #             objects.
    #     """

    #     init_date = self.series.index.max()
    #     forecast = self.aggregate_simulations(self.simulated_df)
    #     forecast.loc[init_date, :] = self.series[init_date]
    #     forecast['Fecha'] = forecast.index
    #     forecast.reset_index(drop=True, inplace=True)
    #     series = self.series.to_frame()
    #     last_date = series.index.max()
    #     last_val = series.loc[last_date].item()
    #     series_name = series.columns.item()
    #     series['Fecha'] = series.index
    #     series.reset_index(drop=True, inplace=True)
    #     temp = series.merge(forecast, how='outer', on='Fecha')
        
    #     # Delimit the beginning and end:
    #     temp = temp[(temp['Fecha']>=start) & (temp['Fecha']<=end)]

    #     # Plot the data:
    #     fig, ax = plt.subplots(figsize=figsize)
    #     ax.plot(temp['Fecha'], temp[series_name], 
    #              color=self.COLORS['hist'])
    #     ax.plot(
    #         temp['Fecha'], 
    #         temp['quant_5'], 
    #         color = self.COLORS['perc_5'],
    #         linestyle = ':',
    #         linewidth = 2,
    #         label = 'Perc 5 - Perc 95'
    #         )
    #     ax.plot(
    #         temp['Fecha'], 
    #         temp['quant_95'], 
    #         color = self.COLORS['perc_95'],
    #         linestyle = ':',
    #         linewidth = 2
    #         )
    #     ax.plot(
    #         temp['Fecha'], 
    #         temp['min'], 
    #         color = self.COLORS['min'],
    #         linestyle = '--',
    #         linewidth = 2,
    #         label = 'Min - Max'
    #         )
    #     ax.plot(
    #         temp['Fecha'], 
    #         temp['max'], 
    #         color = self.COLORS['max'],
    #         linestyle = '--',
    #         linewidth = 2
    #         )
    #     ax.plot(
    #         temp['Fecha'], 
    #         temp['mean'], 
    #         color = self.COLORS['mean'],
    #         linewidth = 2,
    #         label = 'Media'
    #         )
    #     # Set the axis values:
    #     month_fmt = mdates.MonthLocator(interval=3)
    #     ax.xaxis.set_major_locator(month_fmt)
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    #     plt.xticks(rotation=90, ha='right')

    #     # Give label of the last observed value:
    #     local_int = temp.loc[
    #         (temp['Fecha']>=last_date-relativedelta(months=time_space)) &
    #         (temp['Fecha']<=last_date+relativedelta(months=int(time_space))),
    #         series_name
    #         ]
    #     plt.text(
    #         x = last_date-relativedelta(months=time_space),
    #         y = local_int.max(),
    #         s = round(last_val, dec),
    #         color = self.COLORS['hist']
    #         )

    #     dates = temp.loc[temp['Fecha']>last_date, 'Fecha']
    #     dist_u = 1+upper_space
    #     dist_d = 1-lower_space

    #     for date in dates:
    #         if date.month%6==0 and (date-last_date).days>90:

    #             # Find label's valeus:
    #             temp_min = round(temp.loc[temp['Fecha']==date, 'min'].item(),
    #                              dec)
    #             temp_max = round(temp.loc[temp['Fecha']==date, 'max'].item(),
    #                              dec)
    #             temp_mean = round(temp.loc[temp['Fecha']==date, 'mean'].item(),
    #                               dec)
    #             temp_95 = round(
    #                 temp.loc[temp['Fecha']==date, 'quant_95'].item(),
    #                 dec
    #                 )
    #             temp_5 = round(temp.loc[temp['Fecha']==date, 'quant_5'].item(),
    #                            dec)

    #             local_int = temp.loc[
    #                 (temp['Fecha']>=date-relativedelta(months=time_space)) &
    #                 (temp['Fecha']<=date+relativedelta(months=int(time_space)))
    #                 ]

    #             pos_values = local_int.max()

    #             # Put labels in plot:
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['min']*dist_d,
    #                 s = temp_min,
    #                 color = self.COLORS['min']
    #                 )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['max']*dist_u,
    #                 s = temp_max,
    #                 color = self.COLORS['max']
    #                 )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['mean']*dist_u,
    #                 s = temp_mean,
    #                 color = self.COLORS['mean']
    #                 )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['quant_95']*dist_u,
    #                 s = temp_95,
    #                 color = self.COLORS['perc_95']
    #                 )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['quant_5']*dist_d*1.1,
    #                 s = temp_5,
    #                 color = self.COLORS['perc_5']
    #                 )

    #     bottom_val = temp.select_dtypes(include='float64').min().min()
    #     top_val = temp.select_dtypes(include='float64').max().max()

    #     if bottom_val<0:
    #         bottom_val = bottom_val*1.1
    #     else:
    #         bottom_val = bottom_val*0.9

    #     plt.ylim(
    #         bottom = bottom_val, 
    #         top = top_val*1.1
    #         )
    #     plt.xlim(
    #         left = temp['Fecha'].min(), 
    #         right = temp['Fecha'].max()+relativedelta(months=time_space)
    #         )
    #     plt.legend()
    #     plt.show()

    #     return fig, ax

    # def plot_historic_variation(
    #     self, start: Union[str, datetime.datetime]='2000-01-01',
    #     delta_t: int=1, figsize: tuple=(15,10), time_space: int=2,
    #     dec: int=2, upper_space: int=0.1, lower_space: float=0.1
    #     ):
    #     """Plots the historical variations of the interest series.

    #     Args:
    #         start (str, optional): starting date of the plot. Defaults 
    #             to '2000-01-01'. Expected format'%Y-%m-%d'
    #         delta_t (int, optional): delta of time from which the 
    #             variation will be computed. Defaults to 1.
    #         figsize (tuple, optional): figure size. Defaults to (15,10).
    #         time_space (int, optional): number of time periods along the
    #             x-axis from which the labels will be positioned. 
    #             Defaults to 2.
    #         dec (int, optional): number of decimals in the labels.
    #             Defaults to 2.
    #         upper_space (int, optional): determines the position of the
    #             upper labels. Defaults to 0.1.
    #         lower_space (float, optional): determines the position of the
    #             lower labels. Defaults to 0.1.

    #     Returns:
    #         tuple: tuple with mlp.figure.Figure, 
    #             mlp..axes._subplots.AxesSubplot which contain the plot
    #             objects.
    #     """
    #     # Compute variation series and key values:
    #     series = self.variation_series(delta_t=delta_t)[start:]
    #     hist_max = series.max()
    #     max_date = series.idxmax()
    #     hist_min = series.min()
    #     min_date = series.idxmin()
    #     last_date = series.index.max()
    #     start_date = series.index.min()
    #     series_name = series.name
    #     series = series.to_frame()
    #     series['Fecha'] = series.index
    #     series.reset_index(drop=True, inplace=True)
    #     upper_space = upper_space+1
    #     lower_space = 1-lower_space

    #     # Plot variation series with labels:
    #     fig, ax = plt.subplots(figsize=figsize)
    #     ax.plot(
    #         series['Fecha'], 
    #         series[series_name],
    #         color = self.COLORS['hist'],
    #         label = 'Variación'
    #         )
    #     ax.axhline(hist_max, color=self.COLORS['hist_max'], 
    #                linestyle=':', linewidth=2, label='Min - Max')
    #     ax.axhline(hist_min, color=self.COLORS['hist_min'], 
    #                linestyle=':', linewidth=2)
    #     month_fmt = mdates.MonthLocator(interval=12)
    #     ax.xaxis.set_major_locator(month_fmt)
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    #     plt.xticks(rotation=90, ha='right')
        
    #     plt.text(
    #         x = start_date-relativedelta(months=time_space),
    #         y = hist_max*upper_space, 
    #         s = f"{max_date.strftime('%b-%y')}: {round(hist_max, dec)}",
    #         color = self.COLORS['hist_max']
    #         )
    #     plt.text(
    #         x = start_date-relativedelta(months=time_space),
    #         y = hist_min*upper_space, 
    #         s = f"{min_date.strftime('%b-%y')}: {round(hist_min, dec)}",
    #         color = self.COLORS['hist_min']
    #         )
    #     for date in series['Fecha']:
    #         if date.month==last_date.month:
    #             local_int = series.loc[
    #                 (series['Fecha']>=date-relativedelta(months=time_space)) &
    #                 (series['Fecha']<=date+relativedelta(months=time_space)),
    #                 series_name
    #                 ]
    #             val = series.loc[series['Fecha']==date, series_name].item()
    #             if val>0 :
    #                 plt.text(
    #                     x = date-relativedelta(months=time_space),
    #                     y = local_int.max()*upper_space,
    #                     s = round(val, dec),
    #                     color = self.COLORS['hist']
    #                     )
    #             else:
    #                 plt.text(
    #                     x = date-relativedelta(months=time_space),
    #                     y = local_int.min()*lower_space,
    #                     s = round(val, dec),
    #                     color = self.COLORS['hist']
    #                     )
    #     if hist_min<0:
    #         bottom = hist_min*1.2
    #     else:
    #         bottom = hist_min*0.8

    #     plt.xlim(
    #         left = start_date-relativedelta(months=time_space),
    #         right = last_date+relativedelta(months=time_space)
    #         )
    #     plt.ylim(
    #         bottom = bottom,
    #         top = hist_max*1.2
    #         )
    #     plt.legend()
    #     plt.show()

    #     return fig, ax
    
    # def plot_full_variations(
    #     self, start: Union[str, datetime.datetime],
    #     end: Union[str, datetime.datetime], delta_t: int=1,
    #     figsize: tuple=(15,10), time_space: int=2, dec: int=2,
    #     upper_space: int=0.1, lower_space: float=0.1
    #     )->tuple:
    #     """Plots both the historical variations and the forecasted
    #     variations of the series, with its, mean,  cone of confidence 
    #     interval and maximum and minimum.

    #     Args:
    #         start (Union[str, datetime.datetime]): start date of the
    #             plot. Expected format: '%Y-%m-%d'.
    #         end (Union[str, datetime.datetime]):  end date of the
    #             plot. Expected format: '%Y-%m-%d'.
    #         delta_t (int, optional): delta of time from which the
    #             variation will be computed. Defaults to 1.
    #         figsize (tuple, optional): figure size. Defaults to (15,10).
    #         time_space (int, optional): number of time periords along 
    #             the x-axis from which the labels will be positioned. 
    #             Defaults to 2.
    #         dec (int, optional): number of decimals in the labels. 
    #             Defaults to 2.
    #         upper_space (int, optional): determines the position of the
    #             upper labels. Defaults to 0.1.
    #         lower_space (float, optional): determines the position of 
    #             the lower labels. Defaults to 0.1.

    #     Returns:
    #         tuple: tuple with mlp.figure.Figure, 
    #             mlp..axes._subplots.AxesSubplot which contain the plot
    #             objects.
    #     """
    #     # Get variations:
    #     var_series = self.variation_series()
    #     series_name = var_series.name
    #     var_forecast = self.variation_forecast_df(delta_t=delta_t)
    #     var_forecast =  self.aggregate_simulations(var_forecast)

    #     # Get important variables
    #     var_series = var_series[start:]
    #     hist_max = var_series.max()
    #     max_date = var_series.idxmax()
    #     hist_min = var_series.min()
    #     min_date = var_series.idxmin()
    #     last_date = var_series.index[-1]
    #     start_date = var_series.index[0]
    #     last_obs_val = var_series.last('1D').item()
    #     dist_u = 1+upper_space
    #     dist_d = 1-lower_space

    #     # Give the correct format and join the full information of variations:
    #     var_series = var_series.to_frame()
    #     var_series['Fecha'] = var_series.index
    #     var_series.reset_index(drop=True, inplace=True)
    #     var_forecast['Fecha'] = var_forecast.index
    #     var_forecast.reset_index(drop=True, inplace=True)
    #     full_var = var_series.merge(var_forecast, how='outer', on='Fecha')

    #     # Delimit the data's beginning and end:
    #     full_var = full_var[
    #         (full_var['Fecha']>=start) &
    #         (full_var['Fecha']<=end)
    #         ]

    #     # Plot the data:
    #     fig, ax = plt.subplots(figsize=figsize)
    #     ax.plot(full_var['Fecha'], full_var[series_name],
    #             color = self.COLORS['hist'])
    #     ax.plot(
    #         full_var['Fecha'], 
    #         full_var['quant_5'], 
    #         color = self.COLORS['perc_5'],
    #         linestyle = ':',
    #         linewidth = 2,
    #         label = 'Perc 5 - Perc 95'
    #         )
    #     ax.plot(
    #         full_var['Fecha'], 
    #         full_var['quant_95'], 
    #         color = self.COLORS['perc_95'],
    #         linestyle = ':',
    #         linewidth = 2
    #         )
    #     ax.plot(
    #         full_var['Fecha'], 
    #         full_var['min'], 
    #         color = self.COLORS['min'],
    #         linestyle = '--',
    #         linewidth = 2,
    #         label = 'Min - Max'
    #         )
    #     ax.plot(
    #         full_var['Fecha'], 
    #         full_var['max'], 
    #         color = self.COLORS['max'],
    #         linestyle = '--',
    #         linewidth = 2
    #         )
    #     ax.plot(
    #         full_var['Fecha'], 
    #         full_var['mean'], 
    #         color = self.COLORS['mean'],
    #         linewidth = 2,
    #         label = 'Media'
    #         )
    #     ax.axhline(hist_max, color=self.COLORS['hist_max'], 
    #                linestyle=':', linewidth=2, label='Min - Max')
    #     ax.axhline(hist_min, color=self.COLORS['hist_min'], 
    #                linestyle=':', linewidth=2)
        
    #     # Set the axis values:
    #     month_fmt = mdates.MonthLocator(interval=3)
    #     ax.xaxis.set_major_locator(month_fmt)
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    #     plt.xticks(rotation=90, ha='right')

    #     # Give label of the reference values:
    #     local_int = full_var.loc[
    #         (full_var['Fecha']>=last_date-relativedelta(months=time_space)) &
    #         (full_var['Fecha']<=last_date+relativedelta(months=time_space)),
    #         series_name
    #         ]
    #     plt.text(
    #         x = last_date-relativedelta(months=time_space),
    #         y = dist_u*local_int.max(),
    #         s = round(last_obs_val,dec),
    #         color = self.COLORS['hist']
    #         )
    #     plt.text(
    #         x = start_date,
    #         y = prop_label(hist_max), 
    #         s = f"{max_date.strftime('%b-%y')}: {round(hist_max,dec)}",
    #         color = self.COLORS['hist_max']
    #         )
    #     plt.text(
    #         x = start_date,
    #         y = prop_label(hist_min), 
    #         s = f"{min_date.strftime('%b-%y')}: {round(hist_min,dec)}",
    #         color = self.COLORS['hist_min']
    #     )

    #     dates = full_var.loc[full_var['Fecha']>last_date, 'Fecha']
    #     for date in dates:
    #         if date.month%6 == 0 and (date-last_date).days>90:
    #             temp_min = round(
    #                 full_var.loc[full_var['Fecha']==date, 'min'].item(),
    #                 dec
    #             )
    #             temp_max = round(
    #                 full_var.loc[full_var['Fecha']==date, 'max'].item(),
    #                 dec
    #             )
    #             temp_mean = round(
    #                 full_var.loc[full_var['Fecha']==date, 'mean'].item(),
    #                 dec
    #             )
    #             temp_95 = round(
    #                 full_var.loc[full_var['Fecha']==date, 'quant_95'].item(),
    #                 dec
    #             )
    #             temp_5 = round(
    #                 full_var.loc[full_var['Fecha']==date, 'quant_5'].item(),
    #                 dec
    #             )
    #             local_int = full_var.loc[
    #                 (full_var['Fecha']>=date-relativedelta(months=time_space)) &
    #                 (full_var['Fecha']<=date+relativedelta(months=time_space))
    #             ]
    #             pos_values = local_int.max()
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['min']*dist_d,
    #                 s = temp_min,
    #                 color = self.COLORS['min']
    #             )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['max']*dist_u,
    #                 s = temp_max,
    #                 color = self.COLORS['max']
    #             )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['mean']*dist_u,
    #                 s = temp_mean,
    #                 color = self.COLORS['mean']
    #             )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['quant_95']*dist_u,
    #                 s = temp_95,
    #                 color = self.COLORS['perc_95']
    #             )
    #             plt.text(
    #                 x = date-relativedelta(months=time_space), 
    #                 y = pos_values['quant_5']*dist_d,
    #                 s = temp_5,
    #                 color = self.COLORS['perc_5']
    #             )
    #     bottom_val = full_var.select_dtypes(include='float64').min().min()
    #     top_val = full_var.select_dtypes(include='float64').max().max()

    #     if bottom_val<0:
    #         bottom_val = bottom_val*1.1
    #     else:
    #         bottom_val = bottom_val*0.9

    #     plt.ylim(
    #         bottom = bottom_val, 
    #         top = top_val*1.1
    #     )
    #     plt.xlim(
    #         left = full_var['Fecha'].min(), 
    #         right = full_var['Fecha'].max()+relativedelta(months=time_space)
    #     )
    #     plt.legend()
    #     return fig, ax