# Auxialary printing & plotting format functions
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
import bisect
from datetime import datetime, timedelta


from estimators import *
import cstats


# Call with:
# matplotlib.rcParams.update(aux.tex_fonts)
#
tex_fonts = {

    #"text.usetex": True, # may fail
    #"font.family": "arial",
    "font.family": "serif",

    # 11 pt here matches 11pt in tex
    "axes.labelsize": 11,
    "font.size": 11,

    # Different sized than above
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}


def date_labels(dates, N=1):
    """ Get date labels for matplotlib plots

    Args:
        dates : array of datetime objects
        N     : hopping (integer)
    
    Returns:
        labels, positions
    """
    ind = np.rint(np.arange(0, len(dates), N)).astype(int)
    labels = [None] * len(ind)
    for i in range(len(ind)):
        labels[i] = dates[ind[i]].strftime('%d/%m')
    positions  = np.arange(0, len(dates), N)
    
    return labels, positions


def get_datetime(dt, shift=0):
    """
    Get datetimes object arrays.
    
    Args:
        dt       : datestrings array
        shift    : shift (integer)
    Return:
        dt_orig  : datetime object array 
        dt_shift : shifted datetime object array
    """
    dt_orig  = [None] * len(dt)
    dt_shift = [None] * len(dt)

    try:
        for i in range(len(dt_orig)):
            dt_orig[i]  = datetime.strptime(dt[i], '%Y-%m-%d')
            dt_shift[i] = dt_orig[i] + timedelta(days = shift)
    except:
        print(f'get_datetime: Problem with input date')

    if shift == 0:
        return dt_orig
    
    dt_tot = []
    for i in range(np.abs(shift)): # Take up to the shift
        dt_tot.append(dt_shift[i])
    for i in range(len(dt_orig)):
        dt_tot.append(dt_orig[i])

    return dt_orig, dt_shift, dt_tot


def set_fig_size(width=426, fraction=1, aspect='wide'):
    """ Set figure dimensions compatible with latex.

    Args:
        width: document textwidth or columnwidth in pts
        fraction: 1 or 0.5 (for half page width)

    Output:
        dimensions of figure

    Examples:
        width   = 426 # Obtain from your .tex with \showthe\textwidth command
        fig, ax = plt.subplots(1, 1, figsize=set_fig_size(width))
    """

    # Width of figure in pts
    width_pt = fraction * width

    # pts to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio
    if aspect == 'golden':
        ratio = (5**0.5 - 1) / 2
    elif aspect == 'wide':
        ratio = 0.71
    else:
        ratio = aspect

    # in inches
    width_inch  = width_pt * inches_per_pt
    height_inch = ratio * width_inch

    return (width_inch, height_inch)


def set_arr_format(precision):
    """ Set numpy array print format.
    """
    a = '{' + ':.{}f'.format(precision) + '}'
    float_formatter = a.format
    np.set_printoptions(formatter = {'float_kind' : float_formatter })


def printbar(symbol='-', length=80):
    """ Print horizontal ascii bar.
    """
    for i in range(length): print(symbol, end='')
    print('')


def printB(B, counts=False):
    """ Print data information.
    """
    print(' TIF (Tested, Infected, Fatal) triplet combinations')
    printbar()
    for i in range(B.shape[1]):
        binstr = format(i, 'b').zfill(3)
        
        if counts:
            print(' ' + binstr + f' | p[{i}] = {np.mean(B[:,i]):<6.1f} \t Q68: [{np.percentile(B[:,i], Q68[0]):>7.1f}, {np.percentile(B[:,i], Q68[1]):>7.1f}] \t Q95: [{np.percentile(B[:,i], Q95[0]):>7.1f}, {np.percentile(B[:,i], Q95[1]):>7.1f}]')
        else:
            print(' ' + binstr + f' | p[{i}] = {np.mean(B[:,i]):<6.5f} \t Q68: [{np.percentile(B[:,i], Q68[0]):>7.5f}, {np.percentile(B[:,i], Q68[1]):>7.5f}] \t Q95: [{np.percentile(B[:,i], Q95[0]):>7.5f}, {np.percentile(B[:,i], Q95[1]):>7.5f}]')
        
    if counts:
        print(f'     |  sum = {np.sum(np.mean(B, axis=0)):.1f}')
    else:
        print(f'     |  sum = {np.sum(np.mean(B, axis=0)):.5f}')


def pf(name, x, counts=False):
    """ Print formatted variable.
    """
    if counts:
        print(f' {name:>6s} = {np.mean(x):>6.1f} \t Q68: [{np.percentile(x, Q68[0]):>7.1f}, {np.percentile(x, Q68[1]):>7.1f}]  \t Q95: [{np.percentile(x, Q95[0]):>7.1f}, {np.percentile(x, Q95[1]):>7.1f}]')
    else:
        print(f' {name:>6s} = {np.mean(x):>6.4f} \t Q68: [{np.percentile(x, Q68[0]):>7.4f}, {np.percentile(x, Q68[1]):>7.4f}]  \t Q95: [{np.percentile(x, Q95[0]):>7.4f}, {np.percentile(x, Q95[1]):>7.4f}]')


# ------------------------------------------------------------------------
def set_color_cycle(self, clist=None):
    """ Reset matplotlib color cycles.
    """
    if clist is None:
        clist = rcParams['axes.color_cycle']
    self.color_cycle = itertools.cycle(clist)

def set_color_cycle(self, clist):
    """
    Set the color cycle for any future plot commands on this Axes.

    *clist* is a list of mpl color specifiers.
    """
    self._get_lines.set_color_cycle(clist)
    self._get_patches_for_fill.set_color_cycle(clist)
# ------------------------------------------------------------------------
