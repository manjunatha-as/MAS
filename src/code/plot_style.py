"""Common figure style: Times-like serif text at the size of the manuscript.

Figures are drawn at the width they are printed at, so no scaling happens in
LaTeX and every label keeps the same size. Include them with
\\includegraphics[width=\\textwidth]{...} (FULL) or [width=0.49\\textwidth] (HALF).
The manuscript body is 10.9 pt Times on a 6.32 in text block; figure text is
set to 9 pt (labels) and 8 pt (ticks, legends), the usual caption size.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL = 6.3    # inches, \textwidth
HALF = 3.1    # inches, 0.49\textwidth

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman",
                   "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, path_no_ext):
    """Vector PDF for the manuscript plus a 300-dpi PNG for previews."""
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png", dpi=300)
    plt.close(fig)
