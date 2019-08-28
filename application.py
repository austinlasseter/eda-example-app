import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import pickle
import plotly.figure_factory as ff
from plotly import tools

# Define the color palette (17 colors).
Viridis= ['#440154', '#48186a', '#472d7b', '#424086', '#3b528b', '#33638d', '#2c728e', '#26828e', '#21918c', '#1fa088',
          '#28ae80', '#3fbc73', '#5ec962', '#84d44b', '#addc30','#d8e219', '#fde725']
# testdf = pd.read_csv('data/dataset7.gz', compression='gzip', header=0, sep=',', quotechar='"')


# Metrics
pickle_off = open('data/model_metrics7.pkl','rb')
results = pickle.load(pickle_off)

mydata1 = go.Bar(
    x=results.loc['F1 score'].index,
    y=results.loc['F1 score'],
    name=results.index[0],
    marker=dict(color=Viridis[12])
)
mydata2 = go.Bar(
    x=results.loc['Accuracy'].index,
    y=results.loc['Accuracy'],
    name=results.index[1],
    marker=dict(color=Viridis[7])
)
mydata3 = go.Bar(
    x=results.loc['AUC score'].index,
    y=results.loc['AUC score'],
    name=results.index[2],
    marker=dict(color=Viridis[0])
)
mylayout = go.Layout(
    title='Comparison of Possible Models',
    xaxis = dict(title = 'Predictive models'), # x-axis label
    yaxis = dict(title = 'Score'), # y-axis label

)
fig_metrics = go.Figure(data=[mydata1, mydata2, mydata3], layout=mylayout)

# ROC-AUC figure
pickle_off = open('data/FPR.pkl','rb')
FPR = pickle.load(pickle_off)
pickle_off = open('data/TPR.pkl','rb')
TPR = pickle.load(pickle_off)
roc_score=88.1

data = [
    {
      'x':FPR,
      'y':TPR,
      'type':'scatter',
      'mode': 'lines',
      'name': 'AUC: '+str(roc_score)
      },
     {'x':[0,1],
      'y':[0,1],
      'type':'scatter',
      'mode': 'lines',
      'name': 'Baseline Area: 50.0'}]

layout = go.Layout(
    title = 'Receiver Operating Characteristic - Area Under Curve',
)
rocauc_fig = go.Figure(data=data, layout=layout)

## Confusion Matrix
pickle_off = open('data/confu_matrix7.pkl','rb')
cm = pickle.load(pickle_off)
cm_table = ff.create_table(cm)






########
# Device graph_objs# read clean datafile
gapdf = pd.read_csv('data/dataset2.gz', compression='gzip', header=0, sep=',', quotechar='"')

# heatmap: all attributes, correlation
corrs = pd.DataFrame(gapdf[['attribute2', 'attribute3', 'attribute4',
       'attribute5', 'attribute6', 'attribute7', 'attribute9', 'failure']].corr())
data = [go.Heatmap(z=corrs.values.tolist()[::-1],
                   y=corrs.columns.tolist()[::-1],
                   x=corrs.index.tolist(),
                   colorscale='Viridis')]
layout=go.Layout(
        title="Heatmap of attributes and failure")


heatmap = go.Figure(data=data, layout=layout)






###########
# Descriptive Stats

df = pd.read_csv('data/dataset4.gz', compression='gzip', header=0, sep=',', quotechar='"')
# Devices in service
colx='date'
coly='device'
aggdf=df.groupby(colx)[coly].count().reset_index(drop=False)
data = [go.Scatter(x=aggdf[colx],
                   y=aggdf[coly],
                    mode = 'lines',
                   marker=dict(color=Viridis[0])
)]
layout = go.Layout(
    title = f'Number of devices still in service, by date',
    xaxis = dict(title = colx),
    yaxis = dict(title = coly),
    hovermode ='closest'
)
inservice = go.Figure(data=data, layout=layout)
# Failure
# Distribution by device
coly='device'
colx='failure'
aggdf=df.groupby(coly)[colx].sum().reset_index(drop=False)
counts=aggdf[colx].value_counts().values.tolist()[::-1]

data = [go.Bar(y=counts,
               x=['Failed', "Did not fail"],
               marker=dict(color=[Viridis[0], Viridis[10]]),
                   )]

layout = go.Layout(
    title = f'Distribution of {coly} by {colx}',
    xaxis = dict(title = colx),
    yaxis = dict(title = 'Number of devices'),
    width=500,
    height=400,
)
failurerate = go.Figure(data=data, layout=layout)
# Failure Trend
# Trend over time
colx='date'
coly='failure'
aggdf=df.groupby(colx)[coly].sum().reset_index(drop=False)

data = [go.Scatter(x=aggdf[colx],
                   y=aggdf[coly],
                    mode = 'lines',
                   marker=dict(color=Viridis[5])
)]
layout = go.Layout(
    title = f'Distribution of {coly} by {colx}',
    xaxis = dict(title = colx),
    yaxis = dict(title = coly),
    hovermode ='closest'
)
failure_trend = go.Figure(data=data, layout=layout)
# Prefix
# Distribution by device
coly='device'
colx='prefix'
aggdf=df.groupby(coly)[colx].max().reset_index(drop=False)
counts=aggdf[colx].value_counts().sort_index().values.tolist()
labels=aggdf[colx].value_counts().sort_index().index.tolist()

data = [go.Bar(y=counts,
               x=labels,
               marker=dict(color=Viridis[::3]),
                   )]

layout = go.Layout(
    title = f'Distribution of {coly} by {colx}',
    xaxis = dict(title = colx),
    yaxis = dict(title = coly+' count'),

)
prefix = go.Figure(data=data, layout=layout)
# Mean failure by prefix and device
coly='device'
colx='prefix'
colz='failure'
aggdf=df.groupby(coly).max().reset_index(drop=False)
aggdf.groupby(colx)[colz].mean().sort_index()
labels=aggdf.groupby(colx)[colz].mean().sort_index().index.tolist()
means=aggdf.groupby(colx)[colz].mean().sort_index().values.tolist()
means = [100*elem for elem in means]
rounded_means = ['%.2f' % elem for elem in means]

data = [go.Bar(y=rounded_means,
               x=labels,
               marker=dict(color=Viridis[::3]),
                   )]

layout = go.Layout(
    title = f'Mean percent {colz} by {colx} and {coly}',
    xaxis = dict(title = colx),
    yaxis = dict(title = colz+' rate'),

)
prefix_fail = go.Figure(data=data, layout=layout)
# Device device_life
age_ranges = ["{0}-{1}".format(age, age + 24) for age in range(0, 300, 25)]

coly='device'
colx='ndays'
colz='device life'
aggdf=df.groupby(coly)[colx].mean().reset_index(drop=False)
aggdf[colz] = pd.cut(x=aggdf[colx], bins=12, labels=age_ranges)
counts=aggdf[colz].value_counts().sort_index().values.tolist()
labels=aggdf[colz].value_counts().sort_index().index.tolist()

data = [go.Bar(y=counts,
               x=labels,
               marker=dict(color=Viridis[::1]),
                   )]

layout = go.Layout(
    title = f'Distribution of {colz} in days',
    xaxis = dict(title = 'Days that device is active'),
    yaxis = dict(title = coly+' count'),
#     width=500,
#     height=400,
)
device_life = go.Figure(data=data, layout=layout)
# Device life by failure
colx='device'
coly='ndays'
colz='failure'
aggdf=df.groupby(colx)[[coly,colz]].mean().reset_index(drop=False)
aggdf['failed']=0
aggdf.loc[aggdf['failure']>0, 'failed']=1
means=aggdf.groupby('failed')['ndays'].mean().values.tolist()
rounded_means= ['%.2f' % elem for elem in means]
labels=['Did not fail', 'Failed']

data = [go.Bar(y=rounded_means,
               x=labels,
               marker=dict(color=Viridis[::10]),
                   )]

layout = go.Layout(
    title = f'Average device life by {colx} and {colz}',
    xaxis = dict(title = colz),
    yaxis = dict(title = 'Days that device is active'),
    width=500,
    height=400,
)
device_life_fail = go.Figure(data=data, layout=layout)
# Distribution of Recoded Attribute 5
data = [go.Histogram(x=df['att5'],
                     xbins=dict(size=1),
                     marker=dict(color=Viridis[::1]+Viridis[::-1]+Viridis[::1]+Viridis[::-1]))]
layout = go.Layout(
    title = 'Attribute 5',
    xaxis = dict(title ='Range of codes in Attribute (0-100)'),
    yaxis = dict(title ='Number of observations'),
)
att5_dist = go.Figure(data=data, layout=layout)
# Attribute 5 by failure
trace0 = go.Box(
    y=df[df['failure']==0]['att5'],
    name = 'Did not fail',
    marker = dict(color=Viridis[0])
)
trace1 = go.Box(
    y=df[df['failure']==1]['att5'],
    name = 'Failed',
    marker = dict(color=Viridis[14])
)
layout = go.Layout(
    title = 'Attribute 5 by Device Failure',
    xaxis = dict(title ='Failure'),
    yaxis = dict(title ='Attribute 5 code (from 1-100)')
)
data=[trace0, trace1]
att5_fail = go.Figure(data=data, layout=layout)
# Attribute 6
transformed6=df['att6']*.01
data = [go.Histogram(x=transformed6,
                     xbins=dict(size=100),
                     marker=dict(color=Viridis[::1]+Viridis[::-1]+Viridis[::1]+Viridis[::-1]+Viridis[::1]+Viridis[::-1]+Viridis[::1]+Viridis[::-1]))]
layout = go.Layout(
    title = 'Attribute 6',
    xaxis = dict(title ='Range of codes in Attribute (0-2500)'),
    yaxis = dict(title ='Number of observations'),
)
att6_dist = go.Figure(data=data, layout=layout)
# Att 6 failure
# Attribute 6 by failure
trace0 = go.Box(
    y=df[df['failure']==0]['att6'],
    name = 'Did not fail',
    marker = dict(color=Viridis[0])
)
trace1 = go.Box(
    y=df[df['failure']==1]['att6'],
    name = 'Failed',
    marker = dict(color=Viridis[14])
)
layout = go.Layout(
    title = 'Attribute 6 by Device Failure',
    xaxis = dict(title ='Failure'),
    yaxis = dict(title ='Attribute 6 code (from 0-665K)')
)
data=[trace0, trace1]
att6_fail = go.Figure(data=data, layout=layout)
# Binary attributers
# Attributes 2, 3, 4, and 7
labels=['zero value', 'all other values']
values2=df['attribute2'].value_counts().values.tolist()
values3=df['attribute3'].value_counts().values.tolist()
values4=df['attribute4'].value_counts().values.tolist()
values7=df['attribute7'].value_counts().values.tolist()

trace2 = go.Bar(x=labels,
               y=values2,
               marker=dict(color=[Viridis[0], Viridis[2]])
              )
trace3 = go.Bar(x=labels,
               y=values3,
               marker=dict(color=[Viridis[4], Viridis[6]])
              )
trace4 = go.Bar(x=labels,
               y=values4,
               marker=dict(color=[Viridis[8], Viridis[10]])
              )
trace7 = go.Bar(x=labels,
               y=values7,
               marker=dict(color=[Viridis[12], Viridis[16]])
              )

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Attribute 2', 'Attribute 3', 'Attribute 4', 'Attribute 7'))
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace7, 2, 2)

fig['layout'].update(height=700,  title='Binary Attributes 2, 3, 4, and 7')
binary_atts=fig
# Binary atts by device failure
# devices that failed had a higher mean for the binary attribute
aggdf=df.groupby('device')[['attribute2', 'attribute3', 'attribute4', 'attribute7', 'failure']].mean()
aggdf['failed']=0
aggdf.loc[aggdf['failure']>0, 'failed']=1
results=aggdf.groupby('failed')['attribute2', 'attribute3', 'attribute4', 'attribute7'].mean()
results
# Graph binary results
mydata1 = go.Bar(
    x=results.loc[0].index,
    y=results.loc[0],
    name='Did not fail',
    marker=dict(color=Viridis[0])
)
mydata2 = go.Bar(
    x=results.loc[1].index,
    y=results.loc[1],
    name='Failed',
    marker=dict(color=Viridis[10])
)

mylayout = go.Layout(
    title='Binary attributes by device failure',
    xaxis = dict(title = 'Binary attributes (0=code zero, 1=all other codes)'), # x-axis label
    yaxis = dict(title = 'Mean value'), # y-axis label

)
binary_failure = go.Figure(data=[mydata1, mydata2], layout=mylayout)









########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# server = app.server
application = app.server
app.title='eda-example'







app.layout = html.Div(children=[
    html.H1(children='Predicting Device Failure'),
html.Div(children='''
        A neural network model for predictive maintenance
    '''),
html.Div(children='''
Challenge: A large fleet of devices requires maintenance to prevent device failure. This repository presents a predictive analysis to help with device maintenance by predicting failure given a series of device attributes, measured daily over the course of 11 months in 2015. The final analysis employs a neural network model with a F1 Score of .96 and a ROC-AUC score of .86.

Dataset: 124,164 daily readings from 1163 devices across 9 attributes related to device failure.

'''),
# Model comparison
html.H3('Model Comparison'),
dcc.Graph(
        id='fig_metrics',
        figure=fig_metrics
    ),
# ROCAUC curve
html.Div([
    html.Div(
        [html.H3('Receiver Operating Characteristic - Area Under Curve'),
        dcc.Graph(
                id='rocauc_fig',
                figure=rocauc_fig
            ),
            ], className='six columns'),
    html.Div([
        html.H3('Confusion Matrix'),
        dcc.Graph(
                id='cm_table',
                figure=cm_table
            ),
        ], className='six columns'),
], className='twelve columns'),
# heatmap
html.H3('Correlation of Device Attributes and Failure'),
dcc.Graph(
        id='heatmap',
        figure=heatmap
    ),


# Devices in service
html.H3('Number of devices still in service, by date'),
dcc.Graph(
        id='inservice',
        figure=inservice
    ),
# Failure rate
html.H3('Number of devices still in service, by date'),
dcc.Graph(
        id='failurerate',
        figure=failurerate
    ),
# Failure trend
html.H3('Trend in failure over time'),
dcc.Graph(
        id='failure_trend',
        figure=failure_trend
    ),
# Prefix
html.Div([
    html.Div(
        [html.H3('Distribution of prefix'),
        dcc.Graph(
                id='prefix',
                figure=prefix
            ),
            ], className='six columns'),
    html.Div([
        html.H3('Prefix by device failure'),
        dcc.Graph(
                id='prefix_fail',
                figure=prefix_fail
            ),
        ], className='six columns'),
], className='twelve columns'),
# Device Life
html.Div([
    html.Div(
        [html.H3('Distribution of device life'),
        dcc.Graph(
                id='device_life',
                figure=device_life
            ),
            ], className='six columns'),
    html.Div([
        html.H3('Device life by device failure'),
        dcc.Graph(
                id='device_life_fail',
                figure=device_life_fail
            ),
        ], className='six columns'),
], className='twelve columns'),
# Attribute 5
html.Div([
    html.Div(
        [html.H3('Distribution of Attribute 5'),
        dcc.Graph(
                id='att5_dist',
                figure=att5_dist
            ),
            ], className='six columns'),
    html.Div([
        html.H3('Attribute 5 by device failure'),
        dcc.Graph(
                id='att5_fail',
                figure=att5_fail
            ),
        ], className='six columns'),
], className='twelve columns'),
# Attribute 6
html.Div([
    html.Div(
        [html.H3('Distribution of Attribute 6'),
        dcc.Graph(
                id='att6_dist',
                figure=att6_dist
            ),
            ], className='six columns'),
    html.Div([
        html.H3('Attribute 6 by device failure'),
        dcc.Graph(
                id='att6_fail',
                figure=att6_fail
            ),
        ], className='six columns'),
], className='twelve columns'),
# Binary attributes
html.H3('Binary attributes: 2, 3, 4, and 7'),
dcc.Graph(
        id='binary_atts',
        figure=binary_atts
    ),
# Binary attributes by failure
html.H3('Binary attributes by device failure'),
dcc.Graph(
        id='binary_failure',
        figure=binary_failure
    ),



])



if __name__ == '__main__':

    application.run(debug=True, port=8080)
