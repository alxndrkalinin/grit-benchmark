from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator



def x_axis_formatter(x, pos, n_labels=5):
    if pos == 1 or pos == n_labels - 1:
        return f'{int(x)}'  # Format the first tick label as '0' without decimals
    else:
        return f'{x:.2f}'  # Use general format for other labels

def plot_map_x3(df, col, title, row=None, move_legend="lower right",
                aspect=None, adjust=None, s=50, pr_x=0.61, pr_y=0.35, l_x=1.05, l_y=0.575, m_x=0.52, m_y=0.01):
    
    unique_col_values = df[col].unique()
    num_cols = len(unique_col_values)
    num_rows = 1
    unique_row_values = [None]

    if row is not None and row in df.columns:
        unique_row_values = df[row].unique()
        num_rows = len(unique_row_values)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), layout='constrained', sharex=True, sharey=True)

    # Adjust for single subplot to avoid indexing error
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])  # Make it 2D for consistent indexing
    elif num_rows == 1 or num_cols == 1:
        axes = np.array([axes])  # Wrap in another array for consistent indexing

    for row_i, row_value in enumerate(unique_row_values):

        row_df = df[df[row] == row_value] if row is not None else df
        row_axes = axes[row_i] if num_rows > 1 else axes[0]

        for col_i, col_value in enumerate(unique_col_values):
            sub_df = row_df[row_df[col] == col_value]
            mean_map = sub_df['mAP'].mean()
            fr = sub_df['p < 0.05'].mean()

            ax_scatter = row_axes[col_i] if num_cols > 1 else row_axes[0]  # Adjust for single column
            if aspect is not None:
                ax_scatter.set(aspect=aspect)
            ax_kde = ax_scatter.inset_axes([0, 1, 1, 0.15], sharex=ax_scatter)
            
            sns.scatterplot(
                ax=ax_scatter,
                data=sub_df, 
                x='mAP', 
                y='-log10(mAP p-value)', 
                hue='p < 0.05',
                s=s
            )
            ax_scatter.set_xlabel("")
            ax_scatter.set_xlim(-0.05, 1.05)
            ax_scatter.set_ylim(-0.1, max(sub_df['-log10(mAP p-value)'])+0.25)
            ax_scatter.xaxis.set_major_locator(MultipleLocator(base=0.25))
            ax_scatter.get_xaxis().set_major_formatter(FuncFormatter(partial(x_axis_formatter, n_labels=6)))

            ax_scatter.set_title(f"{col_value}", fontsize=16, pad=20)

            ## Specific for Fig 3A
            # ax_scatter.set_ylabel(f"Preprocessing: {row_value}\n-log10(mAP p-value)")
            
            if col_i == 1:
                ax_scatter.text(pr_x, 0.02, f"Retrieved: {fr:.0%}", transform=ax_scatter.transAxes)
            else:
                ax_scatter.text(pr_x, pr_y, f"Retrieved: {fr:.0%}", transform=ax_scatter.transAxes)
            
            # if col_i == num_cols - 1 and row_i == 0:
            #     sns.move_legend(ax_scatter, "upper left", bbox_to_anchor=(1., .5), frameon=False)
            # else:
            #     ax_scatter.get_legend().remove()
            handles, labels = ax_scatter.get_legend_handles_labels()
            ax_scatter.get_legend().remove()

            max_kde_y = 0
            for p_value in sorted(sub_df['p < 0.05'].unique()):
                sns.kdeplot(
                    ax=ax_kde,
                    data=sub_df[sub_df['p < 0.05'] == p_value],
                    x='mAP', 
                    label=str(p_value)
                )
                max_kde_y = max(max_kde_y, max(ax_kde.lines[-1].get_ydata()))

            ax_kde.axvline(mean_map, color='grey', linestyle='--')
            if mean_map < 0.5:
                ax_kde.text(mean_map + 0.05, 0.7, f"Mean mAP: {mean_map:.2f}", transform=ax_kde.transAxes)
            else:
                ax_kde.text(mean_map - 0.2, 0.7, f"Mean mAP: {mean_map:.2f}", transform=ax_kde.transAxes)

            ax_kde.get_yaxis().set_visible(False)
            ax_kde.get_xaxis().set_visible(False)
            sns.despine(ax=ax_kde, left=True, bottom=True)

    fig.text(m_x, m_y, 'mAP', ha='center', va='center')

    plt.tight_layout()
    fig.legend(handles, labels, title="p < 0.05", loc="upper center", bbox_to_anchor=(l_x, l_y), frameon=False)
    if adjust is not None:
        fig.subplots_adjust(**adjust)
    # fig.subplots_adjust(right=0.85)
    plt.show()