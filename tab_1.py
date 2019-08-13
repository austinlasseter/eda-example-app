import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import pickle
import plotly.figure_factory as ff

# Define the color palette (17 colors).
Viridis= ['#440154', '#48186a', '#472d7b', '#424086', '#3b528b', '#33638d', '#2c728e', '#26828e', '#21918c', '#1fa088',
          '#28ae80', '#3fbc73', '#5ec962', '#84d44b', '#addc30','#d8e219', '#fde725']
df = pd.read_csv('data/dataset7.gz', compression='gzip', header=0, sep=',', quotechar='"')

pickle_off = open('data/model_metrics7.pkl','rb')
results = pickle.load(pickle_off)


# Metrics
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






app = dash.Dash()
application = app.server
app.layout = html.Div(children=[
    html.H1(children='Predicting Device Failure'),
html.Div(children='''
        A neural network model for predictive maintenance
    '''),

html.H3('Model Comparison'),
dcc.Graph(
        id='fig_metrics',
        figure=fig_metrics
    ),
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
])






app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':

    application.run(debug=True, port=8080)
