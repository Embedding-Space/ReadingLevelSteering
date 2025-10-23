"""
Generate LaTeX Table for Model Statistics

Reads the compiled model statistics and outputs LaTeX table code.
"""

import pandas as pd
from pathlib import Path

# Read the CSV we generated earlier
csv_path = Path('model_regression_stats.csv')
df = pd.read_csv(csv_path)

print("Generating LaTeX table...\n")
print("="*80)
print()

# Generate LaTeX table
latex = r"""\begin{table}[h]
\centering
\caption{Linear Regression Analysis of Reading-Level Steering Across Model Architectures}
\label{tab:model_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Parameters} & \textbf{R²} & \textbf{p-value} & \textbf{Slope} & \textbf{Intercept} & \textbf{n} \\
\hline
"""

for _, row in df.iterrows():
    # Format p-value for LaTeX
    p_val = row['p-value']
    if '<' in p_val:
        # Handle <1×10⁻¹⁵ format
        p_latex = r'$<10^{-15}$'
    elif '×10^' in p_val or 'x10^' in p_val:
        # Handle scientific notation like 2.13×10^-4
        parts = p_val.replace('×', 'x').split('x10^')
        mantissa = parts[0]
        exponent = parts[1]
        p_latex = f'${mantissa} \\times 10^{{{exponent}}}$'
    else:
        # Regular number
        p_latex = row['p-value']

    latex += f"{row['Model']} & {row['Parameters']} & {row['R²']} & {p_latex} & {row['Slope']} & {row['Intercept']} & {row['n']} \\\\\n"

latex += r"""\hline
\end{tabular}
\end{table}
"""

print(latex)
print()
print("="*80)
print("\nCopy the above LaTeX code into your document!")
print("\nNote: Requires amsmath or similar package for proper rendering of exponents.")

# Also save to file
output_path = Path('model_table.tex')
with open(output_path, 'w') as f:
    f.write(latex)

print(f"\n✓ Also saved to {output_path}")
