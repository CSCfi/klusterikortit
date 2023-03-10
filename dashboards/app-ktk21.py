import json, os
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output

import plotly.express as px
import plotly.graph_objects as go

import dash_bootstrap_components as dbc

from dash_bootstrap_templates import load_figure_template
load_figure_template("materia")

from yamlconfig import read_config

# ----------------------------------------------------------------------

config = read_config(verbose=True)
c = config[config["dataset"]]

datadate = c['datadate']
datasets = c['datasets']
maindataset = c['maindataset']

aineisto = "Kouluterveyskysely 2021"
nclusters = 11

datadir = datasets[maindataset]['dir']

dfcc = pd.read_csv('{}/kmeans_centers-{}.csv'.format(datadir, datadate),
                   sep=";", index_col=0)

with open('{}/descriptions-pca-{}.json'.format(datadir, datadate)) as json_file:
    descriptions = json.load(json_file)
with open('{}/dimensions-{}.json'.format(datadir, datadate)) as json_file:
    dimensions = json.load(json_file)
with open('{}/variables-{}.json'.format(datadir, datadate)) as json_file:
    variables = json.load(json_file)

dflist, dfvarslist, dfvlist, dfbglist, dfpcalist = [], [], [], [], []
for ds in datasets:
    _df =  pd.read_csv('{}/klusterit-{}.csv'.format(ds['dir'], datadate))
    _df['description'] = descriptions['clusters']
    _df['longdescription'] = descriptions['cluster_descriptions']
    dflist.append(_df)

    dfvarslist.append(pd.read_csv('{}/klusterimuuttujat2-{}.csv'.format(ds['dir'],
                                                                        datadate),
                                  sep=";"))
    dfvlist.append(pd.read_csv('{}/kmeans_muuttujat-{}.csv'.format(ds['dir'],
                                                                   datadate),
                               sep=";", index_col=0))
    dfbglist.append(pd.read_csv('{}/kmeans_taustamuuttujat-{}.csv'.format(ds['dir'],
                                                                          datadate),
                                sep=";", index_col=0))

    dfpcalist.append(pd.read_csv('{}/pca-{}.csv'.format(ds['dir'], datadate)))

df = dflist[maindataset]

intromd = ""
introfn = '{}/intro.md'.format(datadir)
if os.path.exists(introfn):
    with open(introfn, 'r') as f:
        intromd = f.read() #.replace('\n', '')

pos_color = "#519b2f" # "rgba(33, 150, 243, 1.0)" #"#2196f3"
pos2_color = "#7bc143" # "rgba(33, 150, 243, 0.6)"
neg_color = "#2f61ad" #"rgba(236, 112, 99, 1.0)" # "#ec7063"
neg2_color = "#28a0c1" # "rgba(236, 112, 99, 0.6)"
bgr_color = "#e5e7e9"  # "paleturquoise"
var_color = "#303030" # "#bd3d70"
#bgr_color = "rgb(248, 248, 255)"
zerolinewidth = 3

# ----------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA,
    "https://fonts.googleapis.com/css2?family=Source+Sans+Pro"])

server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    dbc.Row([dbc.Col(html.Img(src='assets/logos.png', style={'height':'60%', 'width':'60%'}),
                     style={"padding": "1rem 1rem"},
                     width=4),
             dbc.Col(html.Div(#'Klusterikortit – '+aineisto,
                              aineisto,
                              style={'font-size': 'xx-large', 'font-family': "Source Sans Pro, sans-serif",
                              'text-align': 'center', "padding": "1rem 1rem"}),
                     width=4,),
            dbc.Col(dcc.Dropdown(id="selected_dataset",
                                 options=[{'label': ds['name'], "value": i} for i, ds in enumerate(datasets)],
                                 value="0", clearable=False, disabled=False if len(datasets)>1 else True,
                                 style={"color": "black"},
                                ), width=2,),
            dbc.Col(dcc.Dropdown(id="selected_page",
                                 options=[{"label": "Aloitussivu", "value": "/"},
                                          {"label": "Klusterikortit", "value": "/page-1"},
                                          {"label": "Pääkomponentit", "value": "/page-2"},
                                          {"label": "Hyvinvoinnin malli", "value": "/page-3"}],
                                 value="/", clearable=False, style={"color": "black"},
                                ), width=2,),
             ], align="center", style={"color": "black", "background-color": "white",
                                       "padding": "0.1rem 0.5rem",
                                       'font-family': "'Source Sans Pro', sans-serif", }),
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
        return dbc.Container(
            dbc.Row(dbc.Col(dcc.Markdown(intromd),
                            width={"size": 10, "offset": 0},)),
            style={"padding": "1rem 2rem",
                   'font-family': "'Source Sans Pro', sans-serif", })
    elif pathname == "/page-1":
        return dbc.Container([
            dbc.Row([dbc.Col(dcc.RadioItems([{'label': '{}: "{}"'.format(int(r['klusteri']),
                                                              r['description']),
                                              'value': int(r['klusteri'])} for _, r in df.iterrows()],
                                            value=1, id='selected_cluster',
                                            labelStyle={'display': 'block'}, inputStyle={"margin-right": "5px"}), width=3),
                     dbc.Col(html.Div(id='longdescription',
                                      style={'font-size': 'x-large', "padding": "1rem 1rem",
                                             "background-color": bgr_color}),
                             width={"size": 9, "offset": 0},),
                     ], align="center", style={"padding": "1rem 1rem", }),
            dbc.Row([dbc.Col([html.Div(id='fractiontitle', style={'font-size': 'large', }),
                              html.Div(id='fraction',
                                       style={'font-size': 'xx-large', "padding": "0.5rem 0.5rem"}),
                              html.Div("Elämään tyytyväisyys:", style={
                                  'font-size': 'large', }),
                              html.Div(id='lifecontentment',
                                       style={'font-size': 'xx-large', "padding": "0.5rem 0.5rem"}),
                              html.Div("Sukupuoli (pojat/tytöt):", style={
                                  'font-size': 'large', }),
                              html.Div(id='sex',
                                       style={'font-size': 'x-large', "padding": "0.5rem 0.5rem"}),
                              dcc.Checklist(options=[{'label': 'Näytä muuttujat', 'value': 'showvars'}],
                                            value=['showvars'], id='show_variables',
                                            inputStyle={"margin-right": "5px"},
                                            style={"padding": "0.5rem 0.5rem"}),
                              dbc.Tooltip("Osuus kaikista nuorista, jotka kuuluvat tähän klusteriin",
                                          target="fraction", placement='bottom-start'),
                              dbc.Tooltip("Klusterin keskimääräinen tyytyväisyysindeksi (maksimiarvo: 1, minimi: 0)",
                                          target="lifecontentment", placement='bottom-start'),
                              dbc.Tooltip("Sukupuolijakauma tässä klusterissa",
                                          target="sex", placement='bottom-start'),
                              dbc.Tooltip("Näytetäänkö yksittäiset muuttujat kullekin hyvinvoinnin ulottuvuudelle",
                                          target="show_variables", placement='bottom-start')
                              ], width=3, style={"padding": "4rem 2rem", }),
                     dbc.Col(dcc.Graph(id="stiglitz"), width=9)]),
            dbc.Row([dbc.Col(dcc.Graph(id="largest_vars"), width=3),
                    dbc.Col(dcc.Graph(id="smallest_vars"), width=3),
                    dbc.Col(dcc.Graph(id="pca"), width=6),
                     ], style={"padding": "2rem 0rem", }),
        ], fluid=True, style={"padding": "1rem 2rem",
                              'font-family': "'Source Sans Pro', sans-serif", })

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
        image_path = 'assets/ktk-malli.png'
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
    df2['order']=range(len(df2))
    dfa = df2.iloc[::2].copy()
    dfb = df2.iloc[1::2].copy()
    dfa["color"] = np.where(dfa['data'] < 0, neg_color, pos_color)
    dfb["color"] = np.where(dfb['data'] < 0, neg2_color, pos2_color)
    df2 = pd.concat([dfa, dfb]).sort_values('order')
    df2['description'] = df2.index.to_series().apply(
        lambda x: dimensions[x]['description'])
    df2['description2']=df2.index+"<br>"+df2.description
    fig = go.Figure(data=go.Bar(x=np.arange(nclusters), y=np.tanh(df2.data.astype(float)),
                                marker=dict(
                                    color=df2.color,
                                    #line=dict(color=df2.color, width=2)
                                    ),
                                customdata=df2.description2,
                                hovertemplate='%{y:.2f}: %{customdata}<extra></extra>'))

    if len(showvars)>0:
        dfvars = dfvarslist[int(ds)]
        dfvarscl = dfvars[dfvars['klusteri']==cl]
        fig.add_trace(go.Scatter(x=dfvarscl['x_noise'],
                                 y=np.tanh(dfvarscl['y']),
                                 marker_color=var_color,
                                 customdata=dfvarscl['variable'],
                                 hovertemplate='%{y:.2f}: %{customdata}<extra></extra>',
                                 mode = 'markers'))

    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis = dict(tickmode = 'array',
                                   tickvals = np.arange(nclusters),
                                   ticktext = df2.index))
    fig.update_yaxes(range=[-1.05, 1.05])
    fig.update_yaxes(zeroline=True, zerolinewidth=zerolinewidth, zerolinecolor='Black')
    fig.update_layout(title={'text':'<b>Hyvinvoinnin ulottuvuudet</b>', 'y':0.93, 'x':0, 'font_size':24})
    fig.update_layout(font_family="'Source Sans Pro', sans-serif")
    return fig

@app.callback(
    Output("fractiontitle", "children"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"))
def update_fractiontitle(cl, ds):
    df2 = dflist[int(ds)]
    sums = df2.sum()
    df2 = df2[df2.klusteri == cl]
    if 'koko' in df2.columns:
        return f'Osuus ({df2.koko.values[0]}/{round(sums.koko)}): '
    return f'Osuus:'

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
    return 'Klusteri {}: '.format(cl)+df2['longdescription']


@app.callback(
    Output("lifecontentment", "children"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"))
def update_contentment(cl, ds):
    dfv = dfvlist[int(ds)]
    #lc = 1-(dfv.lifecontentment[cl-1]-1)/4
    lc = dfv.lifecontentment[cl-1]/4
    return f'{lc:.2f}'

@app.callback(
    Output("sex", "children"),
    Input("selected_cluster", "value"),
    Input("selected_dataset", "value"))
def update_sex(cl, ds):
    dfbg = dfbglist[int(ds)]
    s = dfbg.sukupuoli[cl-1]-1
    return f'{round((1-s)*100)} % / {round(s*100)} %'
    #return f'{(1-s)*100:.1f} % / {s*100:.1f} %'

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
    df_max = dfcc.loc[cl].sort_values(ascending=False)[:10].sort_values().to_frame()
    df_max.columns = ['data']
    df_max['order']=range(len(df_max))
    dfa = df_max.iloc[::2].copy()
    dfb = df_max.iloc[1::2].copy()
    dfa["color"] = np.where(dfa.data < 0, neg_color, pos_color)
    dfb["color"] = np.where(dfb.data < 0, neg2_color, pos2_color)
    df_max = pd.concat([dfa, dfb]).sort_values('order')
    #colorvec = np.where(largest < 0, neg_color, pos_color)
    #color2vec = np.where(largest < 0, neg2_color, pos2_color)
    if 'short_description' in variables[next(iter(variables))]:
        varlabels = df_max.index.to_series().apply(
            lambda x: variables[x]['short_description'])
    else:
        varlabels = df_max.index
    description = df_max.index.to_series().apply(
        lambda x: variables[x]['description'])
    description = description.apply(brify_string)
    fig = go.Figure(data=go.Bar(y=varlabels, #df_max.index, #[s[:21] for s in largest.index],
                                x=np.tanh(df_max.data), orientation='h',
                                marker=dict(
                                    color=df_max.color,
                                    #line=dict(color=colorvec, width=2)
                                    ),
                                customdata=description,
                                hovertemplate='%{x:.2f}: %{y}:<br>%{customdata}<extra></extra>'))
    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_xaxes(zeroline=True, zerolinewidth=zerolinewidth, zerolinecolor='Black')
    fig.update_yaxes(ticklabelposition="inside")
    fig.update_layout(font_family="'Source Sans Pro', sans-serif")
    fig.update_layout(yaxis={'side': 'right'},
                      title={'text':'<b>Yleistä tässä klusterissa</b>', 'y':0.97, 'x':0, 'font_size':22},
                      margin=dict(
                          t=40, # top margin: 30px, you want to leave around 30 pixels to
                          # display the modebar above the graph.
                          b=10, # bottom margin: 10px
                          l=1, # left margin: 10px
                          r=1, # right margin: 10px
                      )
                      ) #, plot_bgcolor=bgr_color, paper_bgcolor=bgr_color)
    return fig


@app.callback(
    Output("smallest_vars", "figure"),
    Input("selected_cluster", "value"))
def update_smallest_vars(cl):
    df_min = dfcc.loc[cl].sort_values(ascending=False)[-10:].to_frame()
    df_min.columns = ['data']
    df_min['order']=range(len(df_min))
    dfa = df_min.iloc[::2].copy()
    dfb = df_min.iloc[1::2].copy()
    dfa["color"] = np.where(dfa.data < 0, neg_color, pos_color)
    dfb["color"] = np.where(dfb.data < 0, neg2_color, pos2_color)
    df_min = pd.concat([dfa, dfb]).sort_values('order')
    if 'short_description' in variables[next(iter(variables))]:
        varlabels = df_min.index.to_series().apply(
            lambda x: variables[x]['short_description'])
    else:
        varlabels = df_min.index
    description = df_min.index.to_series().apply(
        lambda x: variables[x]['description'])
    description = description.apply(brify_string)
    fig = go.Figure(data=go.Bar(y=varlabels, #df_min.index, #[s[:21] for s in smallest.index],
                                x=np.tanh(df_min.data), orientation='h',
                                marker=dict(color=df_min.color),
                                customdata=description,
                                hovertemplate='%{x:.2f}: %{y}:<br>%{customdata}<extra></extra>'))
    fig.update_xaxes(range=[-1.01, 0.01]) #, tickvals=[-1, -0.75, -0.5, -0.25, 0])
    fig.update_xaxes(zeroline=True, zerolinewidth=zerolinewidth, zerolinecolor='Black')
    fig.update_yaxes(ticklabelposition="inside")
    fig.update_layout(font_family="'Source Sans Pro', sans-serif")
    fig.update_layout(title={'text':'<b>Harvinaista tässä klusterissa</b>', 'y':0.97, 'x':0, 'font_size':22},
                      margin=dict(
                          t=40, # top margin: 30px, you want to leave around 30 pixels to
                          # display the modebar above the graph.
                          b=10, # bottom margin: 10px
                          l=1, # left margin: 10px
                          r=1, # right margin: 10px
                      )
                      ) #, plot_bgcolor=bgr_color, paper_bgcolor=bgr_color)
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
                     title="<b>Pääkomponenttivisualisointi</b>",
                     labels={"pca1": 'PCA 1: '+descriptions['pca_components'][0],
                             "pca2": 'PCA 2: '+descriptions['pca_components'][1]},
                     hover_data={'klusteristr': False,
                                 'pca1': False, 'pca2': False, 
                                 'PCA 1': (':.2f', dfpca.pca1),
                                 'PCA 2': (':.2f', dfpca.pca2)})
    fig.update_traces(marker={'size': datasets[int(ds)]['ps']})
    fig.update_layout(font_family="'Source Sans Pro', sans-serif")
    fig.update_layout(showlegend=False,
                      title={'y':0.97, 'x':0, 'font_size':22},
                      margin=dict(
                          t=40, # top margin: 30px, you want to leave around 30 pixels to
                          # display the modebar above the graph.
                          b=10, # bottom margin: 10px
                          l=1, # left margin: 10px
                          r=1, # right margin: 10px
                      )
                      ) #, paper_bgcolor=bgr_color)
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
                     title="<b>Pääkomponenttivisualisointi, kaikki klusterit</b>",
                     labels={"pca1": "PCA 1: "+descriptions['pca_components'][0],
                             "pca2": "PCA 2: "+descriptions['pca_components'][1],
                             "pca3": "PCA 3: "+descriptions['pca_components'][2],
                             "pca4": "PCA 4: "+descriptions['pca_components'][3]},
                     color="Klusterit", height=900,
                     color_discrete_sequence=px.colors.qualitative.G10,
                     category_orders={"Klusterit": [str(x) for x in np.sort(df['klusteri'].unique())]},
                     hover_data={'Klusterit': False,
                                 'pca1': False, 'pca2': False,
                                 'pca3': False, 'pca4': False, 
                                 'PCA 1': (':.2f', dfpca.pca1),
                                 str(comp): (':.2f', dfpca[pcadict[comp]])})
    fig.update_traces(marker={'size': datasets[int(ds)]['ps']})
    fig.update_layout(font_family="'Source Sans Pro', sans-serif")
    fig.update_layout(title={'y':0.95, 'x':0, 'font_size':24})
    fig.update_layout(showlegend=True, legend= {'itemsizing': 'constant'})
    return fig

# ----------------------------------------------------------------------


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)

# ----------------------------------------------------------------------
