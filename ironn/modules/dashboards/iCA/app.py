# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import flask
import plotly.plotly as py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.tools as tls
import math
import os
from apps import coloranalyzer,camera_effect
import collections
import yaml
import warnings
from datetime import datetime
import re
import argparse
warnings.filterwarnings("ignore")
plt.style.use("tableau-colorblind10")

server = flask.Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, server=server,
                external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv_path", type=str,
                    help="input EDA dataset folder path")
# parser.add_argument("--output_csv_path", type=str,
#                     help="output EDA dataset file path")
args = parser.parse_args()

tab_style = {
    'padding': '10px',
    'fontWeight': 'bold',
    'marginBottom':'20px',
    'fontSize':18,
    'height':'50px',
    'color':'grey'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'backgroundColor': 'rgb(0,87,168)',
    'padding': '10px',
    'color': 'white',
    'marginBottom':"20px",
    'height':'50px',
    'fontSize':20
}

app.layout = html.Div(
    [
        # header
        html.Div([
            html.Span("iCA", className='app-title',
                    style={'color': 'white', 'fontSize': 38,
                           "marginLeft":"10px",'text-align':'center',
                           'padding':'50px 50px 50px 10px'}),
            html.Img(src=app.get_asset_url('QIO-IRONN_logo-white-on-blue.png'),
                    style={'float':'right','width':'150px','borderRadius':'5px 5px 0 0'})
            ],
            style={'backgroundColor': 'rgb(0,87,168)','borderRadius':'5px 5px 0 0'}
        ),

        # tabs
        html.Div([

            dcc.Tabs(
                id="tabs",
                children=[
                    dcc.Tab(label="Color Analysis", value="coloranalyzer_tab",
                            style=tab_style,selected_style=tab_selected_style),
                    dcc.Tab(label="Camera Effect", value="camera_tab",
                            style=tab_style,selected_style=tab_selected_style)
                ],
                value="coloranalyzer_tab",
                style={'fontWeight':'bold'}),
            ]),

        # Tab content
        html.Div(id="tab_content", className="row",
                 style={"margin": "5px 5px 5px 10px"}),
    ],
    className="row",
    style={"marginRight": "20px", "marginLeft":"20px",
    'marginTop':'20px', 'marginBottom':'20px',
    "borderTop":"thin lightgrey solid",
    "borderRadius":"5px",
    'borderBottom': 'thin lightgrey solid',
    'borderLeft': 'thin lightgrey solid',
    'borderRight': 'thin lightgrey solid'},
)

# file = os.path.join("data","complete_joined_df.csv")
file = args.input_csv_path
df = pd.read_csv(file)
dct = dict(collections.Counter(df.Type))
ls_type = []
for rock in dct.keys():
    if dct[rock] > 10:
        ls_type.append(rock)
all_options = {
    'CombinedType': ['CW', 'ORE', "DW"],
    'Type': ls_type}

show_cols = ['file_name', 'Type', 'CombinedType',
             'SkewnessBlue', 'KurtosisBlue', 'MeanPixelBlue',
             'SkewnessGreen', 'KurtosisGreen', 'MeanPixelGreen',
             'SkewnessRed', 'KurtosisRed', 'MeanPixelRed',
             'camera','focal_comb','lens_comb','zoom_comb','et_comb',
             'megapx_comb','pixel_size']

def df_to_table(df):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # Body
        [
            html.Tr(
                [
                    html.Td(df.iloc[i][col])
                    for col in df.columns
                ]
            )
            for i in range(len(df))
        ]
    )

# pure color analysis tab
@app.callback(
    Output('rocktypes_pc', 'options'),
    [Input('tier_pc', 'value')])
def set_rocktype_options(tier):
    return [{'label': i, 'value': i} for i in all_options[tier]]

@app.callback(
    Output('rocktypes_pc', 'value'),
    [Input('tier_pc', 'value')])
def set_rocktype_values(tier):
    if tier == "CombinedType":
        return ["ORE"]
    else:
        return ["HEM"]

@app.callback(
    Output('kdeplot_rock_pc', 'figure'),
    [Input('button-run-pc', 'n_clicks')],
    state=[State('feat_pc', 'value'),
     State('channel_pc', 'value'),
     State('tier_pc', 'value'),
     State('rocktypes_pc','value')
     ])
def update_graph(run_click,feat, channel, tier, rocktypes):
    if run_click:
        try:
            df = pd.read_csv(file)
            colname = [col for col in df.columns if channel in col and feat in col][0]
            groups = df.groupby(tier).groups
            fig,ax = plt.subplots(figsize=(10,4.5))
            ls = [df[colname][groups[x]] for x in rocktypes]
            for i in range(len(ls)):
                ls[i].plot.kde(ax=ax,label=rocktypes[i],linewidth=3)
            plt.title(feat+" on "+channel+" Channel",fontsize=20)
            plt.tick_params(labelsize=15)
            plt.xlabel("Feature value", fontsize=15)
            plt.ylabel("Density", fontsize=15)
            min_val = min([min(x) for x in ls])
            max_val = max([max(x) for x in ls])
            ax.set_xlim([min_val, max_val])
            plotly_fig = tls.mpl_to_plotly(fig)
            plotly_fig["layout"]["showlegend"]=True
            plotly_fig["layout"]["autosize"]=True
            plotly_fig["layout"]["hovermode"]="x"
            plotly_fig["layout"]["legend"]={"font":{"size":"18"}}
            return plotly_fig
        except:
            return {'data':[], 'layout':[]}

@app.callback(
    Output("img_table_pc", "children"),
    [Input('button-generate-pc',"n_clicks")],
    [State('feat_pc', 'value'),
    State('channel_pc', 'value'),
    State("feat_lower_bound_pc", "value"),
    State("feat_upper_bound_pc", "value"),
    State("rocktypes_pc","value"),
    State("tier_pc","value")]
)
def leads_table_callback(generate_button,feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rocks, tier):
    if generate_button:
        df = pd.read_csv(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        tmp = df[df[tier].isin(rocks)]

        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        return df_to_table(res)

@app.callback(
    Output('save-table-textbox-pc', 'children'),
    [Input('button-save-pc', 'n_clicks')],
    [State('userid_pc','value'),
    State('feat_pc', 'value'),
    State('channel_pc', 'value'),
    State("feat_lower_bound_pc", "value"),
    State("feat_upper_bound_pc", "value"),
    State("rocktypes_pc","value"),
    State("tier_pc","value")]
)
def save_current_table(savebutton, userid, feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rocks, tier):
    if savebutton:
        df = pd.read_csv(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        tmp = df[df[tier].isin(rocks)]
        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        curr_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        rock_str = '-'.join(str(e) for e in rocks)
        file_name = userid+"_"+curr_time+"_"+\
                    rock_str+"_"+\
                    str.lower(feat_name)+"_"+\
                    str.lower(channel)+"_"+\
                    feat_lower_bound+"_"+\
                    feat_upper_bound
        if len(res) == 0:
            return "Empty table cannot be saved."
        else:
            res.to_csv(os.path.join("results",file_name+".csv"))
            return "Table "+file_name+".csv" +" saved."

@app.callback(
    Output('rock_ce', 'options'),
    [Input('tier_ce', 'value')])
def set_rock_options(tier):
    return [{'label': i, 'value': i} for i in all_options[tier]]

@app.callback(
    Output('rock_ce', 'value'),
    [Input('tier_ce', 'value')])
def set_rock_values(tier):
    if tier == "CombinedType":
        return "ORE"
    else:
        return "HEM"

@app.callback(
    Output("cam_feat_val_ce","options"),
    [Input("tier_ce","value"),
    Input("rock_ce","value"),
    Input('cam_feat_ce','value')])

def set_cam_feat_options(tier, rock, cam_feat):
    df = pd.read_csv(file)
    dff = df[df[tier] == rock]
    if cam_feat == "Camera Model":
        dct = dict(collections.Counter(dff.camera))
    elif cam_feat == "Focal Length":
        dct = dict(collections.Counter(dff.focal_comb))
    elif cam_feat == "Lens Aperture":
        dct = dict(collections.Counter(dff.lens_comb))
    elif cam_feat == "Exposure Time":
        dct = dict(collections.Counter(dff.et_comb))
    elif cam_feat == "Zoom-in Degree":
        dct = dict(collections.Counter(dff.zoom_comb))
    elif cam_feat == "Megapixel":
        dct = dict(collections.Counter(dff.megapx_comb))
    elif cam_feat == "Pixel Size":
        dct = dict(collections.Counter(dff.pixel_size))
    ls_feat = []
    for item in dct.keys():
        if dct[item] > 10:
            ls_feat.append(item)
    return [{'label': i, 'value': i} for i in ls_feat]

@app.callback(
    Output("cam_feat_val_ce","value"),
    [Input("tier_ce","value"),
    Input("rock_ce","value"),
    Input('cam_feat_ce','value')])
def set_cam_feat_value(tier, rock, cam_feat):
    df = pd.read_csv(file)
    dff = df[df[tier] == rock]
    if cam_feat == "Camera Model":
        return [dff["camera"].unique()[0]]
    elif cam_feat == "Focal Length":
        return [dff["focal_comb"].unique()[0]]
    elif cam_feat == "Lens Aperture":
        return [dff["lens_comb"].unique()[0]]
    elif cam_feat == "Exposure Time":
        return [dff["et_comb"].unique()[0]]
    elif cam_feat == "Zoom-in Degree":
        return [dff["zoom_comb"].unique()[0]]
    elif cam_feat == "Megapixel":
        return [dff["megapx_comb"].unique()[0]]
    elif cam_feat == "Pixel Size":
        return [dff["pixel_size"].unique()[0]]

def plot_graph_cam(df, cam_feat_fn, cam_feat_nm, cl_feat_nm, feat, channel, tier, rock, cam_feat_val):
    try:
        ls = [df[(df[cam_feat_nm]==x)&(df[tier]==rock)][cl_feat_nm] for x in cam_feat_val]
        # for x in cam_feat_val:
        #     ls.append(df[(df[cam_feat]==x)&(df[tier]==rock)][feat_nm])
        fig,ax = plt.subplots(figsize=(10,4.55))
        for i in range(len(ls)):
            ls[i].plot.kde(ax=ax,label=cam_feat_val[i],linewidth=3)
        plt.title(feat+" on "+channel+" Channel"+" of "+rock+" with Different " + cam_feat_fn ,fontsize=20)
        plt.tick_params(labelsize=15)
        plt.xlabel("Feature value", fontsize=15)
        plt.ylabel("Density", fontsize=15)
        min_val = min([min(x) for x in ls])
        max_val = max([max(x) for x in ls])
        ax.set_xlim([min_val, max_val])
        plotly_fig = tls.mpl_to_plotly(fig)
        plotly_fig["layout"]["showlegend"]=True
        plotly_fig["layout"]["autosize"]=True
        plotly_fig["layout"]["hovermode"]="x"
        plotly_fig["layout"]["legend"]={"font":{"size":"18"}}
        return plotly_fig
    except:
        return {'data':[], 'layout':[]}

@app.callback(
    Output('kdeplot_camera_ce', 'figure'),
    [Input('button-run-ce','n_clicks')],
    [State('feat_ce', 'value'),
     State('channel_ce', 'value'),
     State('tier_ce', 'value'),
     State('rock_ce', 'value'),
     State('cam_feat_ce','value'),
     State('cam_feat_val_ce', 'value') # multi-values
     # State('tag_name_ce', 'value'),
     # State('tag_val_ce', 'value')
     ])
def update_graph_camera(run_clicks, feat, channel, tier, rock, cam_feat, cam_feat_val):
    if run_clicks:
        df = pd.read_csv(file)
        feat_name = [col for col in df.columns if channel in col and feat in col][0]
        # df = df[df[tier] == rock]
        if cam_feat == "Camera Model":
            return plot_graph_cam(df, cam_feat,"camera", feat_name,feat, channel, tier, rock, cam_feat_val)
        elif cam_feat == "Focal Length":
            return plot_graph_cam(df, cam_feat,"focal_comb", feat_name,feat, channel, tier, rock, cam_feat_val)
        elif cam_feat == "Lens Aperture":
            return plot_graph_cam(df, cam_feat,"lens_comb", feat_name,feat, channel, tier, rock, cam_feat_val)
        elif cam_feat == "Zoom-in Degree":
            return plot_graph_cam(df, cam_feat,"zoom_comb",feat_name, feat, channel, tier, rock, cam_feat_val)
        elif cam_feat == "Exposure Time":
            return plot_graph_cam(df, cam_feat,"et_comb", feat_name,feat, channel, tier, rock, cam_feat_val)
        elif cam_feat == "Megapixel":
            return plot_graph_cam(df,cam_feat, "megapx_comb",feat_name,feat, channel,  tier, rock, cam_feat_val)
        elif cam_feat == "Pixel Size":
            return plot_graph_cam(df,cam_feat, "pixel_size", feat_name,feat, channel, tier, rock, cam_feat_val)

@app.callback(
    Output("img_table_ce", "children"),
    [Input('button-generate-ce',"n_clicks")],
    [State('feat_ce', 'value'),
    State('channel_ce', 'value'),
    State("feat_lower_bound_ce", "value"),
    State("feat_upper_bound_ce", "value"),
    State("rock_ce","value"),
    State("tier_ce","value"),
    State('cam_feat_ce','value'),
    State('cam_feat_val_ce','value')]
)
def leads_table_callback(generate_button,feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rock, tier,cam_feat, cam_feat_val):
    if generate_button:
        df = pd.read_csv(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        if cam_feat == "Camera Model":
            tmp = df[(df[tier] == rock) & (df["camera"].isin(cam_feat_val))]
        elif cam_feat == "Focal Length":
            tmp = df[(df[tier] == rock) & (df["focal_comb"].isin(cam_feat_val))]
        elif cam_feat == "Lens Aperture":
            tmp = df[(df[tier] == rock) & (df["lens_comb"].isin(cam_feat_val))]
        elif cam_feat == "Exposure Time":
            tmp = df[(df[tier] == rock) & (df["et_comb"].isin(cam_feat_val))]
        elif cam_feat == "Zoom-in Degree":
            tmp = df[(df[tier] == rock) & (df["zoom_comb"].isin(cam_feat_val))]
        elif cam_feat == "Megapixel":
            tmp = df[(df[tier] == rock) & (df["megapx_comb"].isin(cam_feat_val))]
        elif cam_feat == "Pixel Size":
            tmp = df[(df[tier] == rock) & (df["pixel_size"].isin(cam_feat_val))]
        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        return df_to_table(res)

@app.callback(
    Output('save-table-textbox-ce', 'children'),
    [Input('button-save-ce', 'n_clicks')],
    [State('feat_ce', 'value'),
    State('channel_ce', 'value'),
    State("feat_lower_bound_ce", "value"),
    State("feat_upper_bound_ce", "value"),
    State("rock_ce","value"),
    State("tier_ce","value"),
    State('cam_feat_ce','value'),
    State('cam_feat_val_ce','value'),
    State('userid_ce','value')]
)
def save_current_table(savebutton, feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rock, tier,cam_feat, cam_feat_val, userid_ce):
    if savebutton:
        df = pd.read_csv(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        if cam_feat == "Camera Model":
            tmp = df[(df[tier] == rock) & (df["camera"].isin(cam_feat_val))]
        elif cam_feat == "Focal Length":
            tmp = df[(df[tier] == rock) & (df["focal_comb"].isin(cam_feat_val))]
        elif cam_feat == "Lens Aperture":
            tmp = df[(df[tier] == rock) & (df["lens_comb"].isin(cam_feat_val))]
        elif cam_feat == "Exposure Time":
            tmp = df[(df[tier] == rock) & (df["et_comb"].isin(cam_feat_val))]
        elif cam_feat == "Zoom-in Degree":
            tmp = df[(df[tier] == rock) & (df["zoom_comb"].isin(cam_feat_val))]
        elif cam_feat == "Megapixel":
            tmp = df[(df[tier] == rock) & (df["megapx_comb"].isin(cam_feat_val))]
        elif cam_feat == "Pixel Size":
            tmp = df[(df[tier] == rock) & (df["pixel_size"].isin(cam_feat_val))]
        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        curr_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        feat_str = '-'.join(str(e) for e in cam_feat_val)
        # rock_str = '-'.join(str(e) for e in rocks)
        file_name = userid_ce+"_"+curr_time+"_"+\
                    feat_str+"_"+\
                    str.lower(feat_name)+"_"+\
                    str.lower(channel)+"_"+\
                    rock+"_"+\
                    feat_lower_bound+"_"+\
                    feat_upper_bound
        if len(res) == 0:
            return "Empty table cannot be saved."
        else:
            res.to_csv(os.path.join("results",file_name+".csv"))
            return "Table "+file_name+".csv" +" saved."


@app.callback(
    Output("tab_content", "children"),
    [Input("tabs", "value")]
    )
def render_content(tab):
    if tab == "coloranalyzer_tab":
        return coloranalyzer.app.layout
    else:
        return camera_effect.app.layout


if __name__ == "__main__":
    app.run_server(debug=True)
