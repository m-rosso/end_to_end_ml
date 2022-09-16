####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

# pip install plotly==4.6.0
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# Class for creating a grid of plots:

class Plot(ABC):
    """
    Class for creating a grid of plots.

    Arguments for initialization:
        :param grid: number of rows and columns in the grid.
        :type grid: tuple.
        :param titles: titles for each plot in the grid.
        :type titles: list.
        :param width: width of the entire plot.
        :type width: integer.
        :param height: height of the entire plot.
        :type height: integer.
        :param legend: indicates whether to plot legend.
        :type legend: boolean.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, grid: tuple = (1,1), titles: Optional[List[str]] = None,
                 width: int = 900, height: int = 450, legend: bool = True,
                 main_title: Optional[str] = None,
                 template: Optional[str] = None):
        self.grid = grid
        self.fig = make_subplots(rows=self.grid[0], cols=self.grid[1],
                                 subplot_titles=None if titles is None else tuple(titles),
                                 specs=[[{'secondary_y': True}]*self.grid[1]]*self.grid[0])
        self.width = width
        self.height = height
        self.legend = legend
        self.main_title = main_title
        self.template = template
    
    @abstractmethod
    def add_plot(self):
        pass

    def render(self):
        """
        Function for displaying the created grid of plots.
        """
        # Changing layout:
        self.fig.update_layout(
            width=self.width, height=self.height,
            showlegend=self.legend,
            title=self.main_title,
            template=self.template
        )

        # Plotting the figure:
        self.fig.show()

    def export(self, path_to_file):
        """
        Function that saves the created grid of plots.
        """
        self.fig.write_html(path_to_file)

####################################################################################################################################
# Class for creating a grid of lineplots:

class LinePlot(Plot):
    """
    Class for creating a grid of lineplots.

    Methods:
        "add_plot": function that creates a lineplot.
        "render": function for displaying the created grid of plots.
        "export": function that saves the created grid of plots.
    """
    def add_plot(self, data: pd.DataFrame, x: str, y: Union[str, List[str]],
                 position: tuple,
                 legend: bool = True,
                 text_vars: Optional[List[str]] = None,
                 x_axis_name: Optional[str] = None,
                 y_axis_name: Optional[str] = None,
                 y_axis_name_sec: Optional[Union[str, List[str]]] = None,
                 secondary_axis: Optional[List[bool]] = None,
                 colors: Optional[List[str]] = None,
                 styles: Optional[List[str]] = None,
                 line_widths: Optional[List[str]] = None,
                 y_lim: Optional[tuple] = None,
                 y_lim_sec: Optional[tuple] = None) -> None:
        """
        Function that creates a lineplot.

        :param data: data with variables to be plotted.
        :type data: dataframe.
        :param x: name of the variable to be plotted in x-axis.
        :type x: string.
        :param y: list with names of variables to be plotted in y-axis.
        :type y: list of strings.
        :param text_vars: list with names of variables to configure the hovertext.
        :type text_vars: list of strings.
        :param position: position in the grid.
        :type position: tuple.
        :param x_axis_name: name of x-axis.
        :type x_axis_name: string.
        :param y_axis_name: name of (primary) y-axis.
        :type y_axis_name: string.
        :param y_axis_name_sec: name of (secondary) y-axis.
        :type y_axis_name_sec: string.
        :param secondary_axis: list indicating whether each y-variable should be plotted on the secondary axis.
        :type secondary_axis: list of booleans.
        :param colors: list with colors names for each y-variable.
        :type colors: list of strings.
        :param styles: list indicating the style of the line of each y-variable.
        :type styles: list of strings.
        :param line_widths: list indicating the line width of each y-variable.
        :type line_widths: list of strings.
        :param y_lim: limits of y-axis.
        :type y_lim: tuple.
        :param y_lim_sec: limits of secondary y-axis.
        :type y_lim_sec: tuple.

        :return: constructed plot.
        :rtype: None.
        """
        if isinstance(y, str):
            y = [y]
        if isinstance(colors, str):
            colors = [colors]
        if colors is None:
            colors = [None for i in range(len(y))]
        if styles is None:
            styles = [None for i in range(len(y))]
        if line_widths is None:
            line_widths = [None for i in range(len(y))]
        if secondary_axis is None:
            secondary_axis = [False for i in range(len(y))]

        # Defining the plot:
        for p in range(len(y)):
            # Text to be displayed with hover:
            if text_vars is not None:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>' + '%{text}<br>'
                additional_text = [''.join(i) for i in zip(*[[f'{variable} = {v}<br>' for v in data[variable]] for variable in text_vars])]
            else:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>'
                additional_text = None

            self.fig.add_trace(
                go.Scatter(
                    x=data[x], y=data[y[p]], name=y[p],
                    hovertemplate=hovertext, text=additional_text,
                    line=dict(color=colors[p], width=line_widths[p], dash=styles[p]),
                    mode='lines'
                ),
                row=position[0], col=position[1], secondary_y=secondary_axis[p]
            )

        # Changing axes:
        self.fig.update_xaxes(title_text=x if x_axis_name is None else x_axis_name,
                              row=position[0], col=position[1])
        
        if sum(secondary_axis) > 0:
            y_axis = ', '.join([v for v, a in zip(y, secondary_axis) if a==False]) if y_axis_name is None else y_axis_name
            y_axis_sec = ', '.join([v for v, a in zip(y, secondary_axis) if a]) if y_axis_name_sec is None else y_axis_name_sec
            
            self.fig.update_yaxes(title_text=y_axis, row=position[0], col=position[1],
                                  range=None if y_lim is None else y_lim)
            self.fig.update_yaxes(title_text=y_axis_sec, row=position[0], col=position[1], secondary_y=True,
                                  range=None if y_lim_sec is None else y_lim_sec)
        else:
            self.fig.update_yaxes(
                title_text=', '.join(y) if y_axis_name is None else y_axis_name,
                row=position[0], col=position[1],
                range=None if y_lim is None else y_lim
            )

####################################################################################################################################
# Class for creating a grid of scatter plots:

class ScatterPlot(Plot):
    """
    Class for creating a grid of scatter plots.

    Methods:
        "add_plot": function that creates a scatter plot.
        "render": function for displaying the created grid of plots.
        "export": function that saves the created grid of plots.
    """
    def add_plot(self, data: pd.DataFrame, x: str, y: Union[str, List[str]],
                 position: tuple,
                 text_vars: Optional[List[str]] = None,
                 x_axis_name: Optional[str] = None,
                 y_axis_name: Optional[str] = None,
                 y_axis_name_sec: Optional[Union[str, List[str]]] = None,
                 secondary_axis: Optional[List[bool]] = None,
                 colors: Optional[List[str]] = None,
                 y_lim: Optional[tuple] = None,
                 y_lim_sec: Optional[tuple] = None) -> None:
        """
        Function that creates a scatter plot.

        :param data: data with variables to be plotted.
        :type data: dataframe.
        :param x: name of the variable to be plotted in x-axis.
        :type x: string.
        :param y: list with names of variables to be plotted in y-axis.
        :type y: list of strings.
        :param text_vars: list with names of variables to configure the hovertext.
        :type text_vars: list of strings.
        :param position: position in the grid.
        :type position: tuple.
        :param x_axis_name: name of x-axis.
        :type x_axis_name: string.
        :param y_axis_name: name of (primary) y-axis.
        :type y_axis_name: string.
        :param y_axis_name_sec: name of (secondary) y-axis.
        :type y_axis_name_sec: string.
        :param secondary_axis: list indicating whether each y-variable should be plotted on the secondary axis.
        :type secondary_axis: list of booleans.
        :param colors: list with colors names for each y-variable.
        :type colors: list of strings.
        :param y_lim: limits of y-axis.
        :type y_lim: tuple.
        :param y_lim_sec: limits of secondary y-axis.
        :type y_lim_sec: tuple.

        :return: constructed plot.
        :rtype: None.
        """
        if isinstance(y, str):
            y = [y]
        if isinstance(colors, str):
            colors = [colors]
        if colors is None:
            colors = [None for i in range(len(y))]
        if secondary_axis is None:
            secondary_axis = [False for i in range(len(y))]

        # Defining the plot:
        for p in range(len(y)):
            # Text to be displayed with hover:
            if text_vars is not None:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>' + '%{text}<br>'
                additional_text = [''.join(i) for i in zip(*[[f'{variable} = {v}<br>' for v in data[variable]] for variable in text_vars])]
            else:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>'
                additional_text = None

            self.fig.add_trace(
                go.Scatter(
                    x=data[x], y=data[y[p]], name=y[p],
                    hovertemplate=hovertext, text=additional_text,
                    marker_color=colors[p],
                    mode='markers'
                ),
                row=position[0], col=position[1], secondary_y=secondary_axis[p]
            )

        # Changing axes:
        self.fig.update_xaxes(title_text=x if x_axis_name is None else x_axis_name,
                              row=position[0], col=position[1])
        
        if sum(secondary_axis) > 0:
            y_axis = ', '.join([v for v, a in zip(y, secondary_axis) if a==False]) if y_axis_name is None else y_axis_name
            y_axis_sec = ', '.join([v for v, a in zip(y, secondary_axis) if a]) if y_axis_name_sec is None else y_axis_name_sec
            
            self.fig.update_yaxes(title_text=y_axis, row=position[0], col=position[1],
                                  range=None if y_lim is None else y_lim)
            self.fig.update_yaxes(title_text=y_axis_sec, row=position[0], col=position[1], secondary_y=True,
                                  range=None if y_lim_sec is None else y_lim_sec)
        else:
            self.fig.update_yaxes(
                title_text=', '.join(y) if y_axis_name is None else y_axis_name,
                row=position[0], col=position[1]
            )

####################################################################################################################################
# Class for creating a grid of bar plots:

class BarPlot(Plot):
    """
    Class for creating a grid of bar plots.

    Methods:
        "add_plot": function that creates a bar plot.
        "render": function for displaying the created grid of plots.
        "export": function that saves the created grid of plots.
    """
    def add_plot(self, data: pd.DataFrame, x: str, y: Union[str, List[str]],
                 position: tuple,
                 text_vars: Optional[List[str]] = None,
                 x_axis_name: Optional[str] = None,
                 y_axis_name: Optional[str] = None,
                 y_axis_name_sec: Optional[Union[str, List[str]]] = None,
                 secondary_axis: Optional[List[bool]] = None,
                 colors: Optional[List[str]] = None,
                 y_lim: Optional[tuple] = None,
                 y_lim_sec: Optional[tuple] = None) -> None:
        """
        Function that creates a bar plot.

        :param data: data with variables to be plotted.
        :type data: dataframe.
        :param x: name of the variable to be plotted in x-axis.
        :type x: string.
        :param y: list with names of variables to be plotted in y-axis.
        :type y: list of strings.
        :param text_vars: list with names of variables to configure the hovertext.
        :type text_vars: list of strings.
        :param position: position in the grid.
        :type position: tuple.
        :param x_axis_name: name of x-axis.
        :type x_axis_name: string.
        :param y_axis_name: name of (primary) y-axis.
        :type y_axis_name: string.
        :param y_axis_name_sec: name of (secondary) y-axis.
        :type y_axis_name_sec: string.
        :param secondary_axis: list indicating whether each y-variable should be plotted on the secondary axis.
        :type secondary_axis: list of booleans.
        :param colors: list with colors names for each y-variable.
        :type colors: list of strings.
        :param y_lim: limits of y-axis.
        :type y_lim: tuple.
        :param y_lim_sec: limits of secondary y-axis.
        :type y_lim_sec: tuple.

        :return: constructed plot.
        :rtype: None.
        """
        if isinstance(y, str):
            y = [y]
        if colors is None:
            colors = [None for i in range(len(y))]
        if secondary_axis is None:
            secondary_axis = [False for i in range(len(y))]

        # Defining the plot:
        for p in range(len(y)):
            # Text to be displayed with hover:
            if text_vars is not None:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>' + '%{text}<br>'
                additional_text = [''.join(i) for i in zip(*[[f'{variable} = {v}<br>' for v in data[variable]] for variable in text_vars])]
            else:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>'
                additional_text = None

            self.fig.add_trace(
                go.Bar(
                    x=data[x], y=data[y[p]], name=y[p],
                    hovertemplate=hovertext, text=additional_text
                ),
                row=position[0], col=position[1], secondary_y=secondary_axis[p]
            )

        # Changing axes:
        self.fig.update_xaxes(title_text=x if x_axis_name is None else x_axis_name,
                              row=position[0], col=position[1])
        
        if sum(secondary_axis) > 0:
            y_axis = ', '.join([v for v, a in zip(y, secondary_axis) if a==False]) if y_axis_name is None else y_axis_name
            y_axis_sec = ', '.join([v for v, a in zip(y, secondary_axis) if a]) if y_axis_name_sec is None else y_axis_name_sec
            
            self.fig.update_yaxes(title_text=y_axis, row=position[0], col=position[1],
                                  range=None if y_lim is None else y_lim)
            self.fig.update_yaxes(title_text=y_axis_sec, row=position[0], col=position[1], secondary_y=True,
                                  range=None if y_lim_sec is None else y_lim_sec)
        else:
            self.fig.update_yaxes(
                title_text=', '.join(y) if y_axis_name is None else y_axis_name,
                row=position[0], col=position[1]
            )

####################################################################################################################################
# Class for creating a grid of box plots:

class BoxPlot(Plot):
    """
    Class for creating a grid of box plots.

    Methods:
        "add_plot": function that creates a box plot.
        "render": function for displaying the created grid of plots.
        "export": function that saves the created grid of plots.
    """
    def add_plot(self, data: pd.DataFrame, x: str, y: Union[str, List[str]],
                 position: tuple,
                 text_vars: Optional[List[str]] = None,
                 x_axis_name: Optional[str] = None,
                 y_axis_name: Optional[str] = None,
                 y_axis_name_sec: Optional[Union[str, List[str]]] = None,
                 secondary_axis: Optional[List[bool]] = None,
                 colors: Optional[List[str]] = None,
                 y_lim: Optional[tuple] = None,
                 y_lim_sec: Optional[tuple] = None) -> None:
        """
        Function that creates a box plot.

        :param data: data with variables to be plotted.
        :type data: dataframe.
        :param x: name of the variable to be plotted in x-axis.
        :type x: string.
        :param y: list with names of variables to be plotted in y-axis.
        :type y: list of strings.
        :param text_vars: list with names of variables to configure the hovertext.
        :type text_vars: list of strings.
        :param position: position in the grid.
        :type position: tuple.
        :param x_axis_name: name of x-axis.
        :type x_axis_name: string.
        :param y_axis_name: name of (primary) y-axis.
        :type y_axis_name: string.
        :param y_axis_name_sec: name of (secondary) y-axis.
        :type y_axis_name_sec: string.
        :param secondary_axis: list indicating whether each y-variable should be plotted on the secondary axis.
        :type secondary_axis: list of booleans.
        :param colors: list with colors names for each y-variable.
        :type colors: list of strings.
        :param y_lim: limits of y-axis.
        :type y_lim: tuple.
        :param y_lim_sec: limits of secondary y-axis.
        :type y_lim_sec: tuple.

        :return: constructed plot.
        :rtype: None.
        """
        if isinstance(y, str):
            y = [y]
        if colors is None:
            colors = [None for i in range(len(y))]
        if secondary_axis is None:
            secondary_axis = [False for i in range(len(y))]

        # Defining the plot:
        for p in range(len(y)):
            # Text to be displayed with hover:
            if text_vars is not None:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>' + '%{text}<br>'
                additional_text = [''.join(i) for i in zip(*[[f'{variable} = {v}<br>' for v in data[variable]] for variable in text_vars])]
            else:
                hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>'
                additional_text = None

            self.fig.add_trace(
                go.Box(
                    x=None if x is None else data[x], y=data[y[p]], name=y[p],
                    hovertemplate=hovertext, text=additional_text,
                ),
                row=position[0], col=position[1]
            )

        # Changing axes:
        self.fig.update_xaxes(title_text=x if x_axis_name is None else x_axis_name,
                              row=position[0], col=position[1])
        
        if sum(secondary_axis) > 0:
            y_axis = ', '.join([v for v, a in zip(y, secondary_axis) if a==False]) if y_axis_name is None else y_axis_name
            y_axis_sec = ', '.join([v for v, a in zip(y, secondary_axis) if a]) if y_axis_name_sec is None else y_axis_name_sec
            
            self.fig.update_yaxes(title_text=y_axis, row=position[0], col=position[1],
                                  range=None if y_lim is None else y_lim)
            self.fig.update_yaxes(title_text=y_axis_sec, row=position[0], col=position[1], secondary_y=True,
                                  range=None if y_lim_sec is None else y_lim_sec)
        else:
            self.fig.update_yaxes(
                title_text=', '.join(y) if y_axis_name is None else y_axis_name,
                row=position[0], col=position[1]
            )

####################################################################################################################################
# Class for creating a grid of histograms:

class HistoPlot(Plot):
    """
    Class for creating a grid of histograms.

    Methods:
        "add_plot": function that creates a histogram.
        "render": function for displaying the created grid of plots.
        "export": function that saves the created grid of plots.
    """
    def add_plot(self, data: pd.DataFrame, x: str,
                 position: tuple,
                 histnorm: Optional[str] = None,
                 opacity: float = 1.0,
                 x_name: Optional[str] = None,
                 barmode: str = 'overlay',
                 x_axis_name: Optional[str] = None,
                 y_axis_name: Optional[str] = None,
                 color: Optional[str] = None) -> None:
        """
        Function that creates a histogram.

        :param data: data with variables to be plotted.
        :type data: dataframe.
        :param x: name of the variable to be plotted in x-axis.
        :type x: string.
        :param position: position in the grid.
        :type position: tuple.
        :param histnorm: type of samples count in each bin.
        :type histnorm: string.
        :param opacity: degree of opacity for the plot.
        :type opacity: float.
        :param x_name: name of the series.
        :type x_name: string.
        :param x_axis_name: name of x-axis.
        :type x_axis_name: string.
        :param y_axis_name: name of y-axis.
        :type y_axis_name: string.
        :param color: color name for x-variable.
        :type color: string.  

        :return: constructed plot.
        :rtype: None.
        """
        # Defining the plot:
        self.fig.add_trace(
            go.Histogram(
                x=data[x], name=x if x_name is None else x_name,
                histnorm=histnorm,
                marker_color=color
            ),
            row=position[0], col=position[1], secondary_y=False
        )
        self.fig.update_traces(opacity=opacity)

        # Changing axes:
        self.fig.update_xaxes(
            title_text=x if x_axis_name is None else x_axis_name,
            row=position[0], col=position[1]
        )
        
        self.fig.update_yaxes(
            title_text='' if y_axis_name is None else y_axis_name,
            row=position[0], col=position[1]
        )
    
        self.fig.update_layout(
            barmode=barmode
        )

####################################################################################################################################
# Class for creating a grid of heatmaps:

class HeatMap(Plot):
    """
    Class for creating a grid of heatmaps.

    Methods:
        "add_plot": function that creats a heatmap.
        "render": function for displaying the created grid of plots.
        "export": function that saves the grid of plots.
    """

    def add_plot(self, data: list, x: list, y: list,
                 position: tuple,
                 x_axis_name: Optional[str] = None,
                 y_axis_name: Optional[str] = None,
                 numerical_var: Optional[str] = None,
                 x_name: Optional[str] = None,
                 y_name: Optional[str] = None,
                 colorscale: Optional[str] = None,
                 showscale: bool = True) -> None:
        """
        Function that creates a heatmap.

        :param data: data with variables to be plotted.
        :type data: dataframe.
        :param x: list with values for x-axis.
        :type x: list.
        :param y: list with values for y-axis.
        :type y: list.
        :param position: position in the grid.
        :type position: tuple.
        :param x_axis_name: name of x-axis.
        :type x_axis_name: string.
        :param y_axis_name: name of y-axis.
        :type y_axis_name: string.
        :param numerical_var: name of numerical variable.
        :type numerical_var: string.
        :param x_name: name of x variable.
        :type x_name: string.
        :param y_name: name of y variable.
        :type y_name: string.
        :param colorscale: name of color gradient.
        :type colorscale: string.
        :param showscale: indicates whether to show the color scale.
        :type showscale: boolean.

        :return: constructed plot.
        :rtype: None.
        """
        if numerical_var is None:
            numerical_var = 'Z'
        if x_name is None:
            x_name = 'X'
        if y_name is None:
            y_name = 'Y'
        
        self.fig.add_trace(
            go.Heatmap(
                z=data,
                x=x,
                y=y,
                colorscale=colorscale,
                colorbar={'title': numerical_var},
                showscale=showscale,
                hovertemplate=x_name + ': %{x}<br>' + y_name + ': %{y}<br>' + numerical_var + ': %{z}<extra></extra>'
            ),
            row=position[0], col=position[1], secondary_y=False
        )

        # Changing axes:
        self.fig.update_xaxes(side='top', title_text=x_axis_name)
        self.fig.update_yaxes(title_text=y_axis_name)

####################################################################################################################################
# Function that plots a grid of histograms:

def plot_histogram(data: pd.DataFrame, x: List[str], pos: List[Tuple[int]], by_var: Optional[List[str]] = None,
                   histnorm: Optional[str] = None, barmode: str = 'overlay', opacity: float = 0.75,
                   x_title: Optional[List[str]] = None, y_title: Optional[List[str]] = None,
                   titles: Optional[List[str]] = None, width: int = 900, height: int = 450) -> None:
    """
    Function that plots a grid of histograms.

    :param data: data with variables to be plotted.
    :type data: dataframe.
    :param x: list with names of variables to be plotted in x-axis.
    :type x: list of strings.
    :param pos: list with position (tuples with row and column) of each plot in the grid.
    :type pos: list of tuples.
    :param by_var: list with names of variables for the hue.
    :type by_var: list of strings.
    :param histnorm: type of histogram. Check Plotly documentation for the alternatives.
    :type histnorm: string.
    :param barmode: type of plots when more than one histogram is plotted.
    :type barmode: string.
    :param opacity: opacity of plots.
    :type opacity: float.
    :param x_title: name of x-axis.
    :type x_title: list of strings.
    :param y_title: name of y-axis.
    :type y_title: list of strings.
    :param titles: list with titles for each plot in the grid.
    :type titles: list of strings.
    :param width: width of the entire plot.
    :type width: integer.
    :param height: height of the entire plot.
    :type height: integer.

    :return: grid of histograms.
    :rtype: plotly graph.
    """
    rows, cols = max([p[0] for p in pos]), max([p[1] for p in pos])
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=tuple(titles))

    # Defining the plot:
    for p in range(len(x)):
        if by_var is not None:
            [fig.add_trace(
                go.Histogram(
                    x=data[data[by_var[p]]==v][x[p]], name=f'{by_var[p]}={v}', histnorm=histnorm,
                ), row=pos[p][0], col=pos[p][1], secondary_y=False
            ) for v in list(data[by_var[p]].unique())]
        
        else:
            fig.add_trace(
                go.Histogram(
                    x=data[x[p]], name=x[p], histnorm=histnorm
                ), row=pos[p][0], col=pos[p][1], secondary_y=False
            )
        
        fig.update_traces(opacity=opacity)

        if y_title is not None:
            # Changing axes:
            fig.update_yaxes(title_text=y_title[p], row=pos[p][0], col=pos[p][1])
            fig.update_xaxes(title_text=x_title[p], row=pos[p][0], col=pos[p][1])

    # Changing layout:
    fig.update_layout(
        width=width, height=height, barmode=barmode
    )

    fig.show()

####################################################################################################################################
# Function that plots a grid of boxplots:

def plot_boxplot(data: pd.DataFrame, x: Optional[List[str]], y: List[str], pos: List[Tuple[int]],
                 titles: Optional[List[str]] = None, width: int = 900, height: int = 450) -> None:
    """
    Function that plots a grid of boxplots.

    :param data: data with variables to be plotted.
    :type data: dataframe.
    :param x: list with names of variables to be plotted in x-axis.
    :type x: list of strings.
    :param y: list with names of variables to be plotted in y-axis.
    :type y: list of strings.
    :param pos: list with position (tuples with row and column) of each plot in the grid.
    :type pos: list of tuples.
    :param titles: list with titles for each plot in the grid.
    :type titles: list of strings.
    :param width: width of the entire plot.
    :type width: integer.
    :param height: height of the entire plot.
    :type height: integer.

    :return: grid of boxplots.
    :rtype: plotly graph.
    """
    rows, cols = max([p[0] for p in pos]), max([p[1] for p in pos])
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=tuple(titles))

    # Defining the plot:
    for p in range(len(x)):
        fig.add_trace(
            go.Box(
                x=None if x is None else data[x[p]], y=data[y[p]], name=y[p],
            ), row=pos[p][0], col=pos[p][1], secondary_y=False
        )
        
        # Changing axes:
        fig.update_yaxes(title_text=y[p], row=pos[p][0], col=pos[p][1])
        fig.update_xaxes(title_text=None if x is None else x[p], row=pos[p][0], col=pos[p][1])

    # Changing layout:
    fig.update_layout(
        width=width, height=height
    )

    fig.show()

####################################################################################################################################
# Function that plots a grid of barplots:

def plot_bar(data: pd.DataFrame, x: List[str], y: List[str], pos: List[Tuple[int]], text_vars: Optional[List[str]] = None,
             titles: Optional[List[str]] = None, width: int = 900, height: int = 450) -> None:
    """
    Function that plots a grid of barplots.

    :param data: data with variables to be plotted.
    :type data: dataframe.
    :param x: list with names of variables to be plotted in x-axis.
    :type x: list of strings.
    :param y: list with names of variables to be plotted in y-axis.
    :type y: list of strings.
    :param pos: list with position (tuples with row and column) of each plot in the grid.
    :type pos: list of tuples.
    :param titles: list with titles for each plot in the grid.
    :type titles: list of strings.
    :param width: width of the entire plot.
    :type width: integer.
    :param height: height of the entire plot.
    :type height: integer.

    :return: grid of barplots.
    :rtype: plotly graph.
    """
    rows, cols = max([p[0] for p in pos]), max([p[1] for p in pos])
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=tuple(titles))

    # Defining the plot:
    for p in range(len(x)):
        # Text to be displayed with hover:
        if text_vars is not None:
            hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x[p]}' + ' = %{x}<br>' + '%{text}<br>'
            additional_text = [''.join(i) for i in zip(*[[f'{variable} = {v}<br>' for v in data[variable]] for variable in text_vars[p]])]
        else:
            hovertext = f'{y[p]}' + ' = %{y}<br>' + f'{x}' + ' = %{x}<br>'
            additional_text = None

        fig.add_trace(
            go.Bar(
                x=data[x[p]], y=data[y[p]], name=y[p], hovertemplate=hovertext, text=additional_text
            ), row=pos[p][0], col=pos[p][1], secondary_y=False
        )
        
        # Changing axes:
        fig.update_yaxes(title_text=y[p], row=pos[p][0], col=pos[p][1])
        fig.update_xaxes(title_text=x[p], row=pos[p][0], col=pos[p][1])

    # Changing layout:
    fig.update_layout(
        width=width, height=height
    )

    fig.show()
