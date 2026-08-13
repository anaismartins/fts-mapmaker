from matplotlib import rc, rcParams

# common setup for matplotlib
params = {'text.usetex' : True,
          'font.size' : 20,
          'font.family' : 'lmodern',
        #   'text.latex.unicode': True,
          'backend': 'pdf',
          'savefig.dpi': 300, # save figures to 300 dpi
          'axes.labelsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.major.pad': 6,
          'xtick.major.pad': 6,
          'ytick.labelsize': 10,
          }

# use of Sans Serif also in math mode
rc('text.latex', preamble='\\usepackage{sfmath}')

rcParams.update(params)

def cm2inch(cm):
    """Centimeters to inches"""
    return cm *0.393701
