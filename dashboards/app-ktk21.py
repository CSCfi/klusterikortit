import json
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output

import plotly.express as px
import plotly.graph_objects as go

import dash_bootstrap_components as dbc

from dash_bootstrap_templates import load_figure_template
load_figure_template("materia")


# ----------------------------------------------------------------------

aineisto = "KTK 2021"
nclusters = 11
datadirs = ["All-2022-10-18", "Turku-2022-10-18", "Jkl-2022-10-18"]
datadir = "Turku-2022-10-18"
datadate = '2022-10-18'
pca_point_sizes = [2, 5, 5]

dfcc = pd.read_csv('{}/kmeans_centers-{}.csv'.format(datadir, datadate),
                   sep=";", index_col=0)

with open('{}/descriptions-pca-{}.json'.format(datadir, datadate)) as json_file:
    descriptions = json.load(json_file)
with open('{}/dimensions-{}.json'.format(datadir, datadate)) as json_file:
    dimensions = json.load(json_file)
with open('{}/variables-{}.json'.format(datadir, datadate)) as json_file:
    variables = json.load(json_file)

dflist, dfvarslist, dfvlist, dfbglist, dfpcalist = [], [], [], [], []
for dd in datadirs:
    _df =  pd.read_csv('{}/klusterit-{}.csv'.format(dd, datadate))
    _df['description'] = descriptions['clusters']
    _df['longdescription'] = descriptions['cluster_descriptions']
    dflist.append(_df)

    dfvarslist.append(pd.read_csv('{}/klusterimuuttujat2-{}.csv'.format(dd,
                                                                        datadate),
                                  sep=";"))
    dfvlist.append(pd.read_csv('{}/kmeans_muuttujat-{}.csv'.format(dd,
                                                                   datadate),
                               sep=";", index_col=0))
    dfbglist.append(pd.read_csv('{}/kmeans_taustamuuttujat-{}.csv'.format(dd,
                                                                          datadate),
                                sep=";", index_col=0))

    dfpcalist.append(pd.read_csv('{}/pca-{}.csv'.format(dd, datadate)))

    #dffac = pd.read_csv('{}/faktorit-{}.csv'.format(datadir, datadate))
    
df = dflist[1]

pos_color = "rgba(33, 150, 243, 1.0)" #"#2196f3"
pos2_color = "rgba(33, 150, 243, 0.6)" 
neg_color = "rgba(236, 112, 99, 1.0)" # "#ec7063"
neg2_color = "rgba(236, 112, 99, 0.6)"
bgr_color = "#e5e7e9"  # "paleturquoise" # "rgb(248, 248, 255)"

# ----------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    dbc.Row([dbc.Col(html.Img(src='assets/alphalogos.png', style={'height':'60%', 'width':'60%'}),
                     style={"padding": "1rem 1rem"},
                     width=4),
             dbc.Col(html.Div('Klusterikortit - '+aineisto,
                              style={'font-size': 'x-large', 'text-align': 'center',
                                     "padding": "1rem 1rem"}),
                     width=4,),
            dbc.Col(dcc.Dropdown(id="selected_dataset",
                                 options=[{"label": "Koko Suomi", "value": "0"},
                                          {"label": "Turku", "value": "1"},
                                          {"label": "Jyväskylä", "value": "2"}],
                                 value="0", clearable=False, style={"color": "black"},
                                ), width=2,),
            dbc.Col(dcc.Dropdown(id="selected_page",
                                 options=[{"label": "Klusterit", "value": "/"},
                                          {"label": "Pääkomponentit", "value": "/page-2"},
                                          {"label": "Stiglitzin malli", "value": "/page-3"}],
                                 value="/", clearable=False, style={"color": "black"},
                                ), width=2,),
             ], align="center", style={"color": "white", "background-color": pos_color,
                                       "padding": "0.1rem 0.5rem", }),
    html.Hr(),
    html.Div(id='page-content'),
    html.Hr(),
])

@app.callback(Output("url", "pathname"), Input("selected_page", "value"))
def update_url_on_dropdown_change(dropdown_value):
    return dropdown_value

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/":
        return dbc.Container([
            dbc.Row([dbc.Col(dcc.Dropdown(
                id='selected_cluster',
                options=[{'label': 'Klusteri {}: "{}"'.format(int(r['klusteri']),
                                                              r['description']),
                          'value': int(r['klusteri'])} for _, r in df.iterrows()],
                value=df.klusteri[0], clearable=False), width=12),
            ], align="center",),
            dbc.Row([dbc.Col(html.Div(id='longdescription',
                                      style={'font-size': 'x-large', "padding": "1rem 1rem",
                                             "background-color": bgr_color}),
                             width={"size": 10, "offset": 1},),
                     ], align="center", style={"padding": "1rem 1rem", }),
            dbc.Row([dbc.Col(dcc.Graph(id="stiglitz"), width=10),  # 9
                    ### dbc.Col(dcc.Graph(id="sex"), width=3),
                     dbc.Col([html.Div("Osuus:", style={'font-size': 'large', }),
                              html.Div(id='fraction',
                                       style={'font-size': 'xx-large', "padding": "0.5rem 0.5rem"}),
                              html.Div("Tyytyväisyys:", style={
                                  'font-size': 'large', }),
                              html.Div(id='lifecontentment',
                                       style={'font-size': 'xx-large', "padding": "0.5rem 0.5rem"}),
                              html.Div("Sukupuoli (t/p):", style={
                                  'font-size': 'large', }),
                              html.Div(id='sex',
                                       style={'font-size': 'x-large', "padding": "0.5rem 0.5rem"}),
                              dcc.Checklist(options=[{'label': 'Näytä muuttujat', 'value': 'showvars'}],
                                            value=['showvars'], id='show_variables',
                                            inputStyle={"margin-right": "5px"},
                                            style={"padding": "0.5rem 0.5rem"}),
                              ], width=2, style={"padding": "4rem 2rem", }),
                     ]),
            dbc.Row([dbc.Col(dcc.Graph(id="smallest_vars"), width=3),
                    dbc.Col(dcc.Graph(id="largest_vars"), width=3),
                    dbc.Col(dcc.Graph(id="pca"), width=6),
                     ]),
        ], fluid=True, style={"padding": "1rem 2rem", })

    elif pathname == "/page-2":
        return dbc.Container([
            dbc.Row(dbc.Col(dcc.Graph(id="large_pca"))),
            dbc.Row([dbc.Col(dcc.RadioItems([{'label': 'PCA 2: '+descriptions['pca_components'][1], 'value': 'PCA 2'},
                                             {'label': 'PCA 3: '+descriptions['pca_components'][2], 'value': 'PCA 3'},
                                             {'label': 'PCA 4: '+descriptions['pca_components'][3], 'value': 'PCA 4'}],
                                            value='PCA 2', id='selected_component',
                                            labelStyle={'display': 'block'}, inputStyle={"margin-right": "5px"}), width=6),
                     dbc.Col([html.Div('Klusteri {}: "{}"'.format(int(r['klusteri']),
                                                                  r['description']), style={'font-size': 'small', }) for _, r in df.iterrows()])
                     ], style={"padding": "1rem 1rem", },
                    )
        ])
    elif pathname == "/page-3":
        image_path = 'assets/stiglitz_ktk21.png'
        return dbc.Container(
            dbc.Row(dbc.Col(html.Img(src=image_path)))
        )

# ----------------------------------------------------------------------

@app.callback(
    Output("stiglitz", "figure"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"),
    Input("show_variables", "value"))
def update_bar_chart(cl, ds, showvars):
    df2 = dflist[int(ds)] 
    df2 = df2[df2.klusteri == cl].transpose()
    df2 = df2[1:(nclusters+1)]
    df2.columns = ['data']
    df2["color"] = np.where(df2['data'] < 0, neg_color, pos_color)
    df2["color2"] = np.where(df2['data'] < 0, neg2_color, pos2_color)
    df2['description'] = df2.index.to_series().apply(
        lambda x: dimensions[x]['description'])
    fig = go.Figure(data=go.Bar(x=np.arange(nclusters), y=np.tanh(df2.data.astype(float)),
                                marker=dict(
                                    color=df2.color2,
                                    line=dict(color=df2.color, width=2)),
                                customdata=df2.description,
                                hovertemplate='%{y:.2f}: %{customdata}<extra></extra>'))

    if len(showvars)>0:
        dfvars = dfvarslist[int(ds)] 
        dfvarscl = dfvars[dfvars['klusteri']==cl]
        fig.add_trace(go.Scatter(x=dfvarscl['x_noise'],
                                 y=np.tanh(dfvarscl['y']),
                                 marker_color="green",
                                 customdata=dfvarscl['variable'],
                                 hovertemplate='%{y:.2f}: %{customdata}<extra></extra>',
                                 mode = 'markers'))

    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis = dict(tickmode = 'array',
                                   tickvals = np.arange(nclusters),
                                   ticktext = df2.index))
    fig.update_yaxes(range=[-1.05, 1.05])
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_layout(title_text='Hyvinvoinnin ulottuvuudet')

    return fig

@app.callback(
    Output("fraction", "children"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"))
def update_fraction(cl, ds):
    df2 = dflist[int(ds)] 
    df2 = df2[df2.klusteri == cl]
    return f'{round(df2.osuus.values[0]*100)} %'

@app.callback(
    Output("longdescription", "children"),
    Input("selected_cluster", "value"))
def update_longdescription(cl):
    df2 = df[df.klusteri == cl]
    return df2['longdescription']


@app.callback(
    Output("lifecontentment", "children"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"))
def update_contentment(cl, ds):
    dfv = dfvlist[int(ds)] 
    lc = dfv.lifecontentment[cl-1]/4
    return f'{lc:.2f}'

@app.callback(
    Output("sex", "children"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"))
def update_sex(cl, ds):
    dfbg = dfbglist[int(ds)] 
    s = dfbg.sukupuoli[cl-1]-1
    return f'{s*100:.1f} / {(1-s)*100:.1f}'

def brify_string(instr):
    parts = instr.split(" ")
    ret = ""
    linelen = 0
    for part in parts:
        if linelen>0:
            if linelen>20:
                ret += "<br>"
                linelen = 0
            else:        
                ret += " "
                linelen += 1
        ret += part
        linelen += len(part)
    return ret

@app.callback(
    Output("largest_vars", "figure"),
    Input("selected_cluster", "value"))
def update_largest_vars(cl):
    largest = dfcc.loc[cl].sort_values(ascending=False)[:10].sort_values()
    colorvec = np.where(largest < 0, neg_color, pos_color)
    color2vec = np.where(largest < 0, neg2_color, pos2_color)
    description = largest.index.to_series().apply(
        lambda x: variables[x]['description'])
    description = description.apply(brify_string)
    fig = go.Figure(data=go.Bar(y=largest.index, #[s[:21] for s in largest.index],
                                x=np.tanh(largest.values), orientation='h',
                                marker=dict(
                                    color=color2vec,
                                    line=dict(color=colorvec, width=2)),
                                customdata=description,
                                hovertemplate='%{x:.2f}: %{y}:<br>%{customdata}<extra></extra>'))
    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(ticklabelposition="inside")
    fig.update_layout(yaxis={'side': 'right'}, title_text="Klusterin suurimmat muuttujat") #, plot_bgcolor=bgr_color, paper_bgcolor=bgr_color)
    return fig


@app.callback(
    Output("smallest_vars", "figure"),
    Input("selected_cluster", "value"))
def update_smallest_vars(cl):
    smallest = dfcc.loc[cl].sort_values(ascending=False)[-10:]
    colorvec = np.where(smallest < 0, neg_color, pos_color)
    color2vec = np.where(smallest < 0, neg2_color, pos2_color)
    description = smallest.index.to_series().apply(
        lambda x: variables[x]['description'])
    description = description.apply(brify_string)
    fig = go.Figure(data=go.Bar(y=smallest.index, #[s[:21] for s in smallest.index],
                                x=np.tanh(smallest.values), orientation='h',
                                marker=dict(
                                    color=color2vec,
                                    line=dict(color=colorvec, width=2)),
                                customdata=description,
                                hovertemplate='%{x:.2f}: %{y}:<br>%{customdata}<extra></extra>'))
    fig.update_xaxes(range=[-1.01, 0.01]) #, tickvals=[-1, -0.75, -0.5, -0.25, 0])
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(ticklabelposition="inside")
    fig.update_layout(title_text="Klusterin pienimmät muuttujat") #, plot_bgcolor=bgr_color, paper_bgcolor=bgr_color)
    return fig

@app.callback(
    Output("pca", "figure"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"))
def update_pca_scatter(cl, ds):
    dfpca = dfpcalist[int(ds)] 
    dfpca["klusteristr"] = dfpca["klusteri"].astype(str)
    all = list(range(1, 99))
    all.remove(cl)
    dfpca["klusteristr"] = dfpca["klusteristr"].replace([str(x) for x in all],
                                                        '99')
    fig = px.scatter(dfpca, x="pca1", y="pca2",
                     color='klusteristr',
                     color_discrete_map={str(cl): pos_color,
                                         '99': bgr_color},
                     title="Pääkomponenttivisualisointi",
                     labels={"pca1": descriptions['pca_components'][0],
                             "pca2": descriptions['pca_components'][1]})
    fig.update_traces(marker={'size': pca_point_sizes[int(ds)]})
    fig.update_layout(showlegend=False) #, paper_bgcolor=bgr_color)
    return fig

@app.callback(
    Output("large_pca", "figure"),
    Input("selected_component", "value"),
    Input("selected_dataset", "value"))
def update_largepca_scatter(comp, ds):
    dfpca = dfpcalist[int(ds)] 
    dfpca["Klusterit"] = dfpca["klusteri"].astype(str)
    pcadict = {'PCA 1':'pca1', 'PCA 2':'pca2', 'PCA 3':'pca3', 'PCA 4':'pca4'}
    fig = px.scatter(dfpca, x="pca1", y=pcadict[comp],
                     title="Pääkomponenttivisualisointi, kaikki klusterit",
                     labels={"pca1": "PCA 1: "+descriptions['pca_components'][0],
                             "pca2": "PCA 2: "+descriptions['pca_components'][1],
                             "pca3": "PCA 3: "+descriptions['pca_components'][2],
                             "pca4": "PCA 4: "+descriptions['pca_components'][3]},
                     color="Klusterit", height=900,
                     color_discrete_sequence=px.colors.qualitative.G10,
                     category_orders={"Klusterit": [str(x) for x in np.sort(df['klusteri'].unique())]})
    fig.update_traces(marker={'size': pca_point_sizes[int(ds)]})
    fig.update_layout(showlegend=True)
    fig.update_layout(legend= {'itemsizing': 'constant'})
    return fig

# ----------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)

# ----------------------------------------------------------------------
