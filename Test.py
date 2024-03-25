import kaleido
import plotly.graph_objects as go
from kaleido.scopes.plotly import PlotlyScope
from weasyprint import HTML

input_path = 'plots/violin_plots_relative_source_contribution_combined_co2_eq_tot_all_scenarios.html'
output_path = 'plots/violin_plots_relative_source_contribution_combined_co2_eq_tot_all_scenarios.pdf'


HTML(input_path).write_pdf(output_path)
