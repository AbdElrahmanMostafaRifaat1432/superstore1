#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.express as px
import pandas as pd
import datetime
import json
import urllib3
import os
import pytz
import plotly.graph_objs as go
import numpy as np
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
from tokenize import group
from dash import Dash, html, dcc, callback, Input, Output, State


# In[2]:


#! pip install pycountry


# In[3]:


data = pd.read_csv('./ProjSuperstore.csv')


# In[4]:


df = data.copy()
df.head()


# # preprocessing for visualization

# In[5]:


df.info()


# In[6]:


df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])


# In[7]:


df.describe()


# In[8]:


df['year']=df['Order_Date'].dt.year
df['month'] = df['Order_Date'].dt.month


# In[9]:


import pycountry
# Sample data
# data = {'country': ['United States', 'Canada', 'United Kingdom', 'Germany']}
# df = pd.DataFrame(data)

# Function to get country abbreviation
def get_country_abbreviation(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_3
    except AttributeError:
        return None

# Apply the function to create a new column
df['country_code'] = df['Country'].apply(get_country_abbreviation)
df['country_code']


# In[10]:


df['Order Day'] = df['Order_Date'].dt.day

day_of_week_names = {
    0: 'Saturday',
    1: 'Sunday',
    2: 'Monday',
    3: 'Tuesday',
    4: 'Wednesday',
    5: 'Thursday',
    6: 'Friday'
}

df['Order Day of Week'] = df['Order_Date'].dt.dayofweek.apply(lambda x: day_of_week_names[x])


# In[11]:


df


# In[12]:


def return_map(df,store_dropdown2):
    
    if ((store_dropdown2 != None) and (store_dropdown2 != [])):
        
        df = df[df['Country'].isin(store_dropdown2)]
    
    data_sum_by_states = df.groupby(['country_code' , 'Country'])[['Sales', 'Profit' , 'Profit_Margin']].sum()
    data_sum_by_states.reset_index(inplace=True)
    #print(data_sum_by_states)

    map_fig_sales=px.choropleth(data_frame=data_sum_by_states, locations="country_code", color="Profit_Margin",
                             custom_data=['Country' ,'Sales' , 'Profit' ,'Profit_Margin' ],
                             scope='world',
                              color_continuous_scale="Plasma")# scope = continent_dropd , template='plotly_dark'
    
    map_fig_sales.update_layout(margin={"r":0,"t":0,"l":0,"b":0} ) #, height=400, width=600
    map_fig_sales.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Sales: %{customdata[1]}<br>profit: %{customdata[2]}<br>profit_margin: %{customdata[3]}")
    return map_fig_sales


# In[13]:


def return_time(df,store_dropdown2 , choice):
        
    if (store_dropdown2 != []) and (store_dropdown2 != None) :
        df = df[df['Country'].isin(store_dropdown2)]

    
    month_Sales = df.groupby(['year','month'])['Sales'].sum().reset_index()
    month_Profit = df.groupby(['year','month'])['Profit'].sum().reset_index()
    month_Sales['year_month'] = month_Sales['year'].astype(str) + '-' + month_Sales['month'].astype(str)
    month_Profit['year_month'] = month_Profit['year'].astype(str) +'-' + month_Profit['month'].astype(str)
    
    
    
    if choice == 0 :
        fig = px.line(data_frame=month_Sales, x='year_month', y='Sales', color_discrete_sequence=['blue'])

        fig.add_trace(
        px.line(data_frame=month_Profit, x='year_month', y='Profit',  color_discrete_sequence=['red']).data[0]
        )

        fig.data[0].update(showlegend=True,name='Sales')
        fig.data[1].update(showlegend=True,name='Profit')
        fig.update_yaxes(title_text='Value')
        fig.update_layout(title='Sales and Profit Time Analysis' ) #, height=400, width=600
        
    elif choice == 1:
        fig = px.area(month_Sales, x='month', y='Sales', color='year',
              labels={'year': 'Year', 'month': 'Month', 'Sales': 'Total Sales'})

        fig.update_layout(
            #height=400, width=600,
            title_text = 'Distribution of Sales per month ',
           # title_x=0.45,title_font=dict(size=20)
            )
    elif choice ==2:
        # Grouping by year and day of the week and averaging sales
        daily_sales_avg_year = df.groupby(['year', 'Order Day of Week']).agg({'Sales': 'mean'}).reset_index()

        #Creating an interactive line plot using Plotly
        fig = px.line(daily_sales_avg_year, x='Order Day of Week', y='Sales', color='year',
                      title='Average Sales for Each Day of the Week (by Year)',
                      labels={'Order Day of Week': 'Day of the Week', 'Sales': 'Average Sales', 'Order Year': 'Year'})    
    return fig


# # Data preprocessing for model

# In[14]:


df_model = df.copy()
df_model


# In[15]:


# Define the rank for each ship mode
ship_mode_rank = {
    'Same Day': 4,
    'First Class': 3,
    'Second Class': 2,
    'Standard Class': 1
}

df_model['Encoded Ship Mode'] = df_model['Ship_Mode'].map(ship_mode_rank)


# In[16]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Group by Customer ID and aggregate sales-related metrics
customer_data = df_model.groupby('Customer_ID').agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Discount': 'mean',
    'Profit': 'sum',
    'Encoded Ship Mode':  lambda x: x.mode().iloc[0]
}).reset_index()

customer_data


# In[23]:


customer_data2 = customer_data.drop(columns="Customer_ID")
customer_data2


# In[32]:


background = '#f5f0e6' ##eadcf4 #f5f5f5 #f5e6e8 #e6f5f2
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding-top": "12rem ",
    "padding-left": "1rem ",
#     "background-color": background,
}
Tabs_STYLE = {
#     "position": "fixed",
    "right": 0,
    "left": 0,
    "margin-top": "3rem"
#     "background-color": background,
}

# the styles for the main content position it to the right of the sidebar and
CONTENT_STYLE = {
    "background-color": background,
    #"padding-right": "8px",
    #"padding-top": "8px ",
}


# In[36]:


controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": col, "value": col} for col in customer_data2.columns
                    ],
                    value="Sales",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y-variable",
                    options=[
                        {"label": col, "value": col} for col in customer_data2.columns
                    ],
                    value="Quantity",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=3),
            ]
        ),
    ],
    body=True,
    style={"background-color": "#f5f0e6", "border-color": "#f5f0e6", "border-radius": "10px"}
)


# In[ ]:


controls_info = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": col, "value": col} for col in customer_data2.columns
                    ],
                    value="Sales",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y-variable",
                    options=[
                        {"label": col, "value": col} for col in customer_data2.columns
                    ],
                    value="Quantity",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=3),
            ]
        ),
    ],
    body=True,
    style={"background-color": "#f5f0e6", "border-color": "#f5f0e6", "border-radius": "10px"}
)


# In[62]:


external_stylesheets = [ dbc.themes.BOOTSTRAP] # 'https://codepen.io/chriddyp/pen/bWLwgP.css'  
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = dbc.Container(
    [
        html.H1('GLOBAL SUPER STORE SALES ANALYSIS', className="text-center"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select a Store"),
                dcc.Dropdown(
                    id='store_dropdown', 
                    options=[
                        {'label': 'Best Selling & Most Profitable Category', 'value': 'Category'},
                        {'label': 'Best Selling & Most Profitable Sub-Category', 'value': 'Sub_Category'},
                        {'label': 'Most Profitable Customer Segment', 'value': 'Segment'},
                        {'label': 'Preferred Shipping Mode', 'value': 'Ship_Mode'},
                        {'label': 'Most Profitable Region', 'value': 'Region'},
                        {'label': 'Top 10 Cities with highest profit margin', 'value': 'City'},
                        {'label': 'Top 10 Cities with highest number of orders', 'value': 'City2'},
                        {'label': 'Top 10 Selling Products', 'value': 'product'},
                    ],
                    value="Category",
                    clearable=True,
                    className="w-100"
                ),
                html.Br(),
                dcc.Graph(id='bar_graph')
            ], width=6),

            dbc.Col([
                html.Label("Select Countries"),
                dcc.Dropdown(
                    id='store_dropdown2', 
                    options=[{'label': str(country), 'value': str(country)} for country in df['Country'].unique()],
                    value=None,
                    multi=True,
                    placeholder='Choose country',
                    clearable=True,
                    className="w-100"
                ),
                html.Br(),
                dcc.Graph(id='map_graph')
            ], width=6)
        ]),

        html.Br(),  # Add a line break between rows

        dbc.Row([
            dbc.Col(dcc.Graph(id='time_graph1'), width=6),
            dbc.Col(dcc.Graph(id='time_graph2'), width=6)
        ]),

        html.Br(),  # Add a line break between rows

        dbc.Row([
            dbc.Col(dcc.Graph(id='time_graph3'), width=6),
            dbc.Col([
                dbc.Container([
                    html.H1("K-means Clustering"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(controls, md=4),
                        dbc.Col(dcc.Graph(id="cluster-graph" ), md=8  )
                    ], justify="center")
                ], fluid=True)
            ], width=6)
        ] )
    ],
    fluid=True , style= CONTENT_STYLE
)
@app.callback(
    Output("cluster-graph", "figure"),
    [
        Input("x-variable", "value"),
        Input("y-variable", "value"),
        Input("cluster-count", "value"),
    ],
)

def make_graph(x, y, n_clusters):
    # minimal input validation, make sure there's at least one cluster
    km = KMeans(n_clusters=max(n_clusters, 1))
    df = customer_data2.loc[:, [x, y]]
    km.fit(df.values)
    df["cluster"] = km.labels_

    centers = km.cluster_centers_

    data = [
        go.Scatter(
            x=df.loc[df.cluster == c, x],
            y=df.loc[df.cluster == c, y],
            mode="markers",
            marker={"size": 8},
            name="Cluster {}".format(c),
        )
        for c in range(n_clusters)
    ]

    data.append(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"color": "red", "size": 12, "symbol": "diamond"},#marker={"color": "#110", "size": 12, "symbol": "diamond"},
            name="Cluster centers",
        )
    )

    layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

    return go.Figure(data=data, layout=layout)


# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in customer_data2.columns
    ]


# functionality is the same for both dropdowns, so we reuse filter_options
app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
    filter_options
)
app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
    filter_options
)


@app.callback(
    Output(component_id='bar_graph', component_property='figure'),
    Output(component_id='map_graph', component_property='figure'),
    Output(component_id='time_graph1', component_property='figure'),
    Output(component_id='time_graph2', component_property='figure'),
    Output(component_id='time_graph3', component_property='figure'),
    
    Input(component_id='store_dropdown',  component_property='value'),
    Input(component_id='store_dropdown2',  component_property='value')
    )


def update_graph(store_dropdown ,store_dropdown2 ):
    
    store_filedf = df
        
    
    
    if (store_dropdown == None) :
        dff= store_filedf[(store_filedf['Category'] == 'category' )  ] #& (store_filedf['Country'] == 'Country' ) 
        dff = store_filedf.groupby('Category').Profit_Margin.sum().reset_index()    
        pchart=px.pie(data_frame = dff, names = store_dropdown, hole=.3)#,height=400 ,width=600
        return pchart , return_map(store_filedf,store_dropdown2) , return_time(df,store_dropdown2 , 0) , return_time(df,store_dropdown2 , 1) , return_time(df,store_dropdown2 , 2) 
    
    
    elif store_dropdown == 'City' or (store_dropdown == 'City2') or (store_dropdown == 'product') :
        
        if (store_dropdown2 != []) and (store_dropdown2 != None) :
            dff2 = store_filedf[store_filedf['Country'].isin(store_dropdown2)]
        else:
            dff2 = store_filedf

        data_sum_by_states = dff2.groupby(['country_code' , 'Country' , 'City'])[['Profit_Margin' , 'Sales_Count']].sum()
        data_sum_by_states.reset_index(inplace=True)
        
        if store_dropdown == 'City':
            data_sum_by_states = data_sum_by_states.sort_values(by=['Profit_Margin'],ascending=False)
            figr = px.bar(data_sum_by_states[:10], x = 'City', y = 'Profit_Margin',
            title = "Bar graph for profit margin by City" , color = 'City' ) #, width=400 , height=400
            
        elif store_dropdown == 'City2':
            data_sum_by_states = data_sum_by_states.sort_values(by=['Sales_Count'],ascending=False)
            figr = px.bar(data_sum_by_states[:10], x = 'City', y = 'Sales_Count',
            title = "Bar graph for sales by City" , color = 'City' ) #, width=400 , height=400
            
        elif store_dropdown == 'product':
            top_selling_products = dff2.groupby('Product_Name')['Sales'].sum().sort_values(ascending=True).tail(10)
            figr = px.bar(
                top_selling_products,
                x='Sales',
                y=top_selling_products.index, # top_selling_products.index, Product Names as y-axis
                orientation='h',  # Horizontal bar chart
                title='Top Ten Selling Products',
                labels={'Sales': 'Total Sales', 'index': 'Product Name'},  # Custom axis labels
                #width=1300,
                #height=600,  # Adjust height as needed
                #template='plotly_dark'  # Optional: Dark theme
            )

        
        
        return figr , return_map(store_filedf,store_dropdown2) ,return_time(df,store_dropdown2 , 0) , return_time(df,store_dropdown2 , 1) , return_time(df,store_dropdown2 , 2)
    
    
    else:

        if (store_dropdown2 != []) and (store_dropdown2 != None) :
            store_filedf = store_filedf[store_filedf['Country'].isin(store_dropdown2)]
        else:
            store_filedf = store_filedf

    
        data_sum_by_states = store_filedf.groupby(['country_code' , 'Country' , store_dropdown ]).sum()
        data_sum_by_states.reset_index(inplace=True)
       
        if (store_dropdown2 == []) or (store_dropdown2 == None) or (len(store_dropdown2)>1):
            data_sum_by_states = data_sum_by_states.groupby([store_dropdown ]).sum()
            data_sum_by_states.reset_index(inplace=True)
        
        data_sum_by_states['Sales_Count_Normalized'] = data_sum_by_states['Sales_Count'] / data_sum_by_states['Sales_Count'].max()
        data_sum_by_states['Profit_Margin_Normalized'] = data_sum_by_states['Profit_Margin'] / data_sum_by_states['Profit_Margin'].max()
        
        data_sum_by_states = data_sum_by_states.sort_values(by='Sales_Count',ascending=False)

        pchart = px.bar(data_sum_by_states, x = store_dropdown, y = ['Sales_Count_Normalized' , 'Profit_Margin_Normalized']  , barmode="group" ) #, width=800 , height =400 , color=store_dropdown
        
        return pchart , return_map(store_filedf,store_dropdown2) , return_time(df,store_dropdown2 , 0) , return_time(df,store_dropdown2 , 1) , return_time(df,store_dropdown2 , 2)
    
    
    
    


# In[63]:

if __name__ == '__main__':
    app.run_server(debug = True)#use_reloader=True , port = 8053



