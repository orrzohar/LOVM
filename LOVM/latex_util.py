from modelGPT.constants import FEATURE_ORDER_DICT
import numpy as np

def dataset_rank_abalation_latex(df):

    df['idx'] = np.arange(len(df))

    latex = "\\begin{table}[t]\n\centering\n\\resizebox{0.5\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{c|c|c|cc|cc|cc|cc@{}}\n\\toprule\n"
    latex += "&  & \multicolumn{7}{c|}{Features} & \multicolumn{2}{c}{Metrics} \\\\\n"
    latex += "\midrule \n"
    latex += " & Row ID & $a_{\\text{IN}}$ & text-f1 & text-acc1 &  \\superclassSymbl  & \\interSimSymbl & \\intraSimSymbl & \\intraClassCloseSymbl & $\\tau$ ($\\uparrow$) & $R_5$ ($\\uparrow$) \\\\\n"
    latex += "\midrule\n"
    latex += "\\multirow{" + str(len(df)) + "}{*}{\\rotatebox[origin=c]{90}{\\parbox[c]{4cm}{\\centering \\textbf{Dataset Ranking}}}}\n"
    
    for idx, row in df.iterrows():
        latex += to_latex(row)

    latex += "\\bottomrule\n"
    latex += "\end{tabular}}\n"
    latex += "\caption{}\n"
    latex += "\label{}\n"
    latex += "\end{table}\n"

    return latex

def model_rank_abalation_latex(df):

    df['idx'] = np.arange(len(df))

    latex = "\\begin{table}[t]\n\centering\n\\resizebox{0.5\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{c|c|c|cc|cc|cc|cc@{}}\n\\toprule\n"
    latex += "&  & \multicolumn{7}{c|}{Features} & \multicolumn{2}{c}{Metrics} \\\\\n"
    latex += "\midrule \n"
    latex += " & Row ID & $a_{\\text{IN}}$ & text-f1 & text-acc1 &  \\superclassSymbl  & \\interSimSymbl & \\intraSimSymbl & \\intraClassCloseSymbl & $\\tau$ ($\\uparrow$) & $R_5$ ($\\uparrow$) \\\\\n"
    latex += "\midrule\n"
    latex += "\\multirow{" + str(len(df)) + "}{*}{\\rotatebox[origin=c]{90}{\\parbox[c]{4cm}{\\centering \\textbf{Model Ranking}}}}\n"
    
    for idx, row in df.iterrows():
        latex += to_latex(row)

    latex += "\\bottomrule\n"
    latex += "\end{tabular}}\n"
    latex += "\caption{}\n"
    latex += "\label{}\n"
    latex += "\end{table}\n"

    return latex

def model_pred_abalation_latex(df):

    df['idx'] = np.arange(len(df))

    latex = "\\begin{table}[t]\n\centering\n\\resizebox{0.5\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{c|c|c|cc|cc|cc|c@{}}\n\\toprule\n"
    latex += "&  & \multicolumn{7}{c|}{Features} & \multicolumn{1}{c}{Metrics} \\\\\n"
    latex += "\midrule \n"
    latex += " & Row ID & $a_{\\text{IN}}$ & text-f1 & text-acc1 &  \\superclassSymbl  & \\interSimSymbl & \\intraSimSymbl & \\intraClassCloseSymbl & $L_1$ ($\\downarrow$) ) \\\\\n"
    latex += "\midrule\n"
    latex += "\\multirow{" + str(len(df)) + "}{*}{\\rotatebox[origin=c]{90}{\\parbox[c]{4cm}{\\centering \\textbf{Metric Prediction}}}}\n"
    
    for idx, row in df.iterrows():
        latex += to_latex_acc(row)

    latex += "\\bottomrule\n"
    latex += "\end{tabular}}\n"
    latex += "\caption{}\n"
    latex += "\label{}\n"
    latex += "\end{table}\n"

    return latex


def to_latex(row):
     
    latex = f"& {row['idx']+1} & "
    for k in FEATURE_ORDER_DICT.keys():
        if k in row['features']:
            latex += "$\checkmark$ & "
        else:
            latex += "$\\times$ & "
        
    latex += " \cellcolor[HTML]{FFFFED} "
    latex += f"{row['k_tau']:.3f} & "
    latex += " \cellcolor[HTML]{FFFFED} "
    latex += f"{row['acc']:.3f}  "

    latex += "\\\\\n"

    return latex

def to_latex_acc(row):
     
    latex = f"& {row['idx']+1} & "
    for k in FEATURE_ORDER_DICT.keys():
        if k in row['features']:
            latex += "$\checkmark$ & "
        else:
            latex += "$\\times$ & "

    latex += " \cellcolor[HTML]{FFFFED} "
    latex += f"{row['l1']:.3f}  "

    latex += "\\\\\n"

    return latex

def to_latex_grid_search_acc(row):
     
    latex = f"& {row['idx']+1} & "

    latex = f"& {row['idx']+1} & "
    

    latex += " \cellcolor[HTML]{FFFFED} "
    latex += f"{row['l1']:.3f}  "

    latex += "\\\\"

    return latex


def model_main_table(df):
    for col in df.columns:
        print(f'############ {col} #############')
        print('   &  '.join(df[col].index))
        print(' \hspace{-0.4em} & \hspace{-0.9em} '.join([str(round(f, 2)) for f in df[col].values]))


