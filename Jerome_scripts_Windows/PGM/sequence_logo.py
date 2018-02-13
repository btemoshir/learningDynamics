import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt


## Original Reference: https://stackoverflow.com/questions/42615527/sequence-logos-in-matplotlib-aligning-xticks
## Color choice: http://weblogo.threeplusone.com/manual.html

def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def aa_color(letter):
    if letter in ['G','S','T','Y','C']:
        return 'green'
    elif letter in ['Q','N']:
        return 'purple'
    elif letter in ['K','R','H']:
        return 'blue'
    elif letter in  ['D','E']:
        return 'red'
    elif letter in ['A','V','L','I','P','W','F','M']:
        return 'black'
    else:
        return 'yellow'


def build_scores(matrix,epsilon = 1e-4):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    all_scores = []
    for site in range(n_sites):
        conservation = np.log2(20) + (np.log2(matrix[site]+epsilon) * matrix[site]).sum()
        liste = []
        order_colors = np.argsort(matrix[site])
        for c in order_colors:
            liste.append( (list_aa[c],matrix[site,c] * conservation) )
        all_scores.append(liste)
    return all_scores


def build_scores2(matrix):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    epsilon = 1e-4
    all_scores = []
    for site in range(n_sites):
        liste = []
        c_pos = np.nonzero(matrix[site] >= 0)[0]
        c_neg = np.nonzero(matrix[site] < 0)[0]

        order_colors_pos = c_pos[np.argsort(matrix[site][c_pos])]
        order_colors_neg = c_neg[np.argsort(-matrix[site][c_neg])]
        for c in order_colors_pos:
            liste.append( (list_aa[c],matrix[site,c],'+') )
        for c in order_colors_neg:
            liste.append( (list_aa[c],-matrix[site,c],'-') )
        all_scores.append(liste)
    return all_scores





fp = FontProperties(family="Arial", weight="bold")
globscale = 1.35
list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W','Y']
LETTERS = dict([ (letter, TextPath((-0.30, 0), letter, size=1, prop=fp) )   for letter in list_aa] )
COLOR_SCHEME = dict( [(letter,aa_color(letter)) for letter in list_aa] )


def Sequence_logo(matrix, data_type='mean',figsize=(10,3),ylabel = None,epsilon=1e-4,show=True):
    if data_type == 'mean':
        all_scores = build_scores(matrix,epsilon=epsilon)
    elif data_type =='weights':
        all_scores = build_scores2(matrix)
    else:
        print 'data type not understood'
        return -1


    fig, ax = plt.subplots(figsize=figsize)

    x = 1
    maxi = 0
    mini = 0
    for scores in all_scores:
        if data_type == 'mean':
            y = 0
            for base, score in scores:
                if score > 0.01:
                    letterAt(base, x,y, score, ax)
                y += score
            x += 1
            maxi = max(maxi, y)


        elif data_type =='weights':
            y_pos = 0
            y_neg = 0
            for base,score,sign in scores:
                if sign == '+':
                    letterAt(base, x,y_pos, score, ax)
                    y_pos += score
                else:
                    y_neg += score
                    letterAt(base, x,-y_neg, score, ax)
            x += 1
            maxi = max(y_pos,maxi)
            mini = min(-y_neg,mini)

    if data_type == 'weights':
        maxi = max(  maxi, abs(mini) )
        mini = -maxi

    plt.xticks(range(1,x))
    plt.xlim((0, x))
    plt.ylim((mini, maxi))
    plt.xlabel('Site',fontsize=20)
    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Information (bits)'
        elif data_type =='weights':
            ylabel = 'Weights'
    plt.ylabel(ylabel,fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def Sequence_logo_multiple(matrix, data_type='mean',figsize=(10,3),ylabel = None,epsilon=1e-4,ncols=1,show=True,count_from=0,title=None):
    N_plots = matrix.shape[0]
    nrows = int(np.ceil(N_plots/float(ncols)))
    figsize = (figsize[0]*ncols,figsize[1]*nrows)
    fig, ax = plt.subplots(nrows,ncols,figsize=figsize)
    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Information (bits)'
        elif data_type =='weights':
            ylabel = 'Weights'
    if type(ylabel) == str:
        ylabels = [ylabel + ' #%s'%i for i in range(1+count_from,N_plots+count_from)]
    else:
        ylabels = ylabel
    if title is None:
        title = ''
    if type(title) == str:
        titles = [title for _ in range(N_plots)]
    else:
        titles = title
    for i in range(N_plots):
        if data_type == 'mean':
            all_scores = build_scores(matrix[i],epsilon=epsilon)
        elif data_type =='weights':
            all_scores = build_scores2(matrix[i])
        else:
            print 'data type not understood'
            return -1
        if (ncols>1) & (nrows>1):
            col = i%ncols
            row = i/ncols
            ax_ = ax[row,col]
        elif (ncols>1) & (nrows==1):
            ax_ = ax[i]
        elif (ncols==1) & (nrows>1):
            ax_ = ax[i]
        else:
            ax_ = ax

        x = 1
        maxi = 0
        mini = 0
        for scores in all_scores:
            if data_type == 'mean':
                y = 0
                for base, score in scores:
                    if score > 0.01:
                        letterAt(base, x,y, score, ax_)
                    y += score
                x += 1
                maxi = max(maxi, y)


            elif data_type =='weights':
                y_pos = 0
                y_neg = 0
                for base,score,sign in scores:
                    if sign == '+':
                        letterAt(base, x,y_pos, score, ax_)
                        y_pos += score
                    else:
                        y_neg += score
                        letterAt(base, x,-y_neg, score, ax_)
                x += 1
                maxi = max(y_pos,maxi)
                mini = min(-y_neg,mini)

        if data_type == 'weights':
            maxi = max(  maxi, abs(mini) )
            mini = -maxi

        ax_.set_xticks(range(1,x))
        ax_.set_xlim((0, x))
        ax_.set_ylim((mini, maxi))
        ax_.set_xlabel('Site',fontsize=20)
        ax_.set_ylabel(ylabels[i],fontsize=18)
        ax_.set_title(titles[i],fontsize=18)        
        ax_.spines['right'].set_visible(False)
        ax_.spines['top'].set_visible(False)
        ax_.yaxis.set_ticks_position('left')
        ax_.xaxis.set_ticks_position('bottom')
        ax_.tick_params(axis='both', which='major', labelsize=14)
        ax_.tick_params(axis='both', which='minor', labelsize=14)
    plt.tight_layout()
    if show:
        plt.show()
    return fig
