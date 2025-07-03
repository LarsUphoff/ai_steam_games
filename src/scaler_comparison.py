import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from data_preprocessing import power_scaler, quantile_scaler, robust_scaler


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from data_preprocessing import power_scaler, quantile_scaler, robust_scaler


def scaler_comparison(df: pd.DataFrame) -> dict:
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

    df_power = pd.DataFrame(
        power_scaler.fit_transform(df[numeric_columns]),
        columns=numeric_columns
    )
    df_quantile = pd.DataFrame(
        quantile_scaler.fit_transform(df[numeric_columns]),
        columns=numeric_columns
    )
    df_robust = pd.DataFrame(
        robust_scaler.fit_transform(df[numeric_columns]),
        columns=numeric_columns
    )

    scalers_data = {
        'PowerTransformer (Yeo-Johnson)': df_power,
        'QuantileTransformer (Uniform)': df_quantile,
        'RobustScaler': df_robust
    }

    df_power_limited = df_power.iloc[:, :15]
    df_quantile_limited = df_quantile.iloc[:, :15]
    df_robust_limited = df_robust.iloc[:, :15]

    scalers_data_limited = {
        'PowerTransformer (Yeo-Johnson)': df_power_limited,
        'QuantileTransformer (Uniform)': df_quantile_limited,
        'RobustScaler': df_robust_limited
    }

    key_columns = [
        "price",
        "positive",
        "average_playtime_forever",
        "estimated_owners_calculated",
    ]

    figures = {}

    # Distribution histograms
    for col in key_columns:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Original",
                "PowerTransformer (Yeo-Johnson)",
                "RobustScaler",
                "QuantileTransformer (Uniform)",
            ],
            vertical_spacing=0.1,
        )

        fig.add_trace(go.Histogram(x=df[col], name="Original", nbinsx=50, opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=df_power[col], name="Power", nbinsx=50, opacity=0.7), row=1, col=2)
        fig.add_trace(go.Histogram(x=df_robust[col], name="Robust", nbinsx=50, opacity=0.5), row=2, col=1)
        fig.add_trace(go.Histogram(x=df_quantile[col], name="Quantile", nbinsx=50, opacity=0.7), row=2, col=2)

        fig.update_layout(title=f"Distribution Comparison: {col}", height=600, showlegend=True)
        figures[f"distribution_{col}"] = pio.to_json(fig)

    # Correlation heatmaps
    heatmap_fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=list(scalers_data_limited.keys()),
        specs=[[{"type": "heatmap"}],
               [{"type": "heatmap"}],
               [{"type": "heatmap"}]],
        vertical_spacing=0.3
    )

    positions = [(1, 1), (2, 1), (3, 1)]

    for i, (scaler_name, df_scaled) in enumerate(scalers_data_limited.items()):
        corr_matrix = df_scaled.corr()
        row, col = positions[i]
        heatmap_fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                showscale=(i == 0),
                name=scaler_name,
            ),
            row=row, col=col
        )

    heatmap_fig.update_layout(
        title="Correlation Matrix Comparison Across Scalers",
        height=1400,
        width=1000,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    figures["correlation_heatmaps"] = pio.to_json(heatmap_fig)

    # Outlier comparison
    selected_cols = ['price', 'positive']
    for col in selected_cols:
        fig = go.Figure()
        for scaler_name, df_scaled in scalers_data.items():
            fig.add_trace(
                go.Box(
                    y=df_scaled[col],
                    name=scaler_name,
                    boxpoints='outliers'
                )
            )
        fig.update_layout(title=f'Outlier Comparison: {col}', yaxis_title='Scaled Values', height=500)
        figures[f"outliers_{col}"] = pio.to_json(fig)

    return figures

