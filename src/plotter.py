# +
# %load_ext jupyternotify
# %reload_ext autoreload
# %autoreload 2
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import string
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

R_SUN = 696342.0 # km
T_SUN = 5788.0 # Kelvin
L_SUN = 3.846e+26 # W
M_SUN = 1.98840987e+30 # kg
AU = 1.495979e+8 # km

    
PLANET_COLORS = dict(
    Mercury="salmon",
    Earth="dodgerblue",
    Neptune="orchid",
    Jupiter="orange"
)

DOWNLOAD_COMMAND = '''wget "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv" -O "data/confirmed_planets.csv"'''




def set_matplotlib_style_settings(major=7, minor=3, linewidth=1.5, grid=False, topon=True, righton=True):
    rcParams["font.size"] = 30
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = True
    rcParams["axes.labelsize"] = 30
    rcParams["axes.titlesize"] = 30
    rcParams["axes.labelpad"] = 10
    rcParams["axes.linewidth"] = linewidth
    rcParams["axes.edgecolor"] = "black"
    rcParams["xtick.labelsize"] = 25
    rcParams["ytick.labelsize"] = 25
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.major.size"] = major
    rcParams["xtick.minor.size"] = minor
    rcParams["ytick.major.size"] = major
    rcParams["ytick.minor.size"] = minor
    rcParams["xtick.minor.width"] = linewidth
    rcParams["xtick.major.width"] = linewidth
    rcParams["ytick.minor.width"] = linewidth
    rcParams["ytick.major.width"] = linewidth
    rcParams["xtick.top"] = topon
    rcParams["ytick.right"] = righton
    rcParams["axes.grid"] = grid
    rcParams["axes.titlepad"] = 8

    

def load_catalog():
    """
    pl_ratror	Ratio of Planet to Stellar Radius
    pl_bmasse	Planet Mass or Mass*sin(i) [Earth Mass]
    pl_orbper	Orbital Period [days] 
    pl_eqt	Equilibrium Temperature [K]
    pl_orbsmax	Orbit Semi-Major Axis [au])
    st_mass	Stellar Mass [Solar mass]
    st_lum	Stellar Luminosity [log(Solar)]
    st_rad	Stellar Radius [Solar Radius]
    st_teff	Stellar Effective Temperature [K]
      
    
    """
    cat = "data/confirmed_planets.csv"
    if not os.path.isfile(cat):
        subprocess.run([DOWNLOAD_COMMAND], shell=True)
    c = pd.read_csv(cat)
    c = c[["disc_year", "discoverymethod", "pl_bmasse", "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_ratror",  "pl_ratrorerr1", "pl_ratrorerr2", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbsmax", "st_mass" , "st_lum", "st_teff", "st_rad", "pl_eqt"]]
    return c

def load_planet_data():
    d = pd.read_csv('data/planet_data.csv')
    d.index=d.name
    return d

def add_subplot_letter(axes, inside=True, fs=25):
    pos = (.025, 0.9) if inside else (-.2,1.03)
    for n, a in enumerate(axes):
        a.text(*pos, f"({string.ascii_lowercase[n]})", transform=a.transAxes, 
                size=fs, weight='extra bold')
        

#####

def get_year_counts(cat):
    cat = cat.copy()
    data =  cat.groupby('disc_year')['pl_bmasse'].nunique().to_dict()
    data = {int(y):c for y,c in data.items()}
    min_y = min(data.keys())
    data[min_y-1] = 0
    return data
    

def get_method_percents(cat, summary=True):
    """Dict sorted in order of low to highest percent"""
    cat = cat.copy()
    data =  cat.groupby('discoverymethod')['pl_bmasse'].nunique().to_dict()
    data = {m:n for m,n in data.items()}
    new_d = {}
    other = 0
    total_p =  sum(data.values())

    rnd = 1 if summary else 3

    for k,v in data.items():
        v = 100 * (v / total_p)
        if v < 1 and summary:
            other += v
        else:
            new_d[k] = np.round(v, rnd)
        
    if summary:
        new_d['Other'] = np.round(other, rnd)
        
    new_d = dict(sorted(new_d.items(), key=lambda item:item[1]))
    return new_d

def squash_small_numeber_types(cat):
    c = cat.copy()
    method_percents = get_method_percents(c, summary=False)
    for method, percent in method_percents.items():
        if percent < 1.0:
            c.loc[c["discoverymethod"]==method, "discoverymethod"] = "Other"
    return c
    

def get_year_counts_by_method(cat, summary=True):
    c = cat.copy()
    years = list(get_year_counts(c).keys())
    method_percents = get_method_percents(c, summary=False)
    if summary:
        c = squash_small_numeber_types(c)
    method_percents = get_method_percents(c, summary)
    method_counts = {}
    for method in method_percents.keys():
        sub = c[c["discoverymethod"]==method]
        year_counts = get_year_counts(sub)
        method_counts[method] = {y:year_counts.get(y,0) for y in years}
    data = pd.DataFrame(method_counts)
    data = data.sort_index()
    return data


def add_year_data_to_bar_plt(ax, year, order, df, label={}):
    d = df.loc[year].to_dict()
    d = {m:d[m] for m in order}
    base = 0.
    colors = {m:f"C{i}" for i,m in enumerate(df.columns.values)}
    for meth, v in d.items():
        kwgs = dict(bottom=base, width=1, color=colors[meth])
        if len(label)>0:
            kwgs['label'] = label[meth]

        ax.bar(year, v, **kwgs)
        base+=v

def plot_counts_per_year(summary=True):
    cat = load_catalog()
    set_matplotlib_style_settings(major=10, minor=5, linewidth=1.5)
    fig, ax = plt.subplots(1,1, figsize = (9, 5))
    d = get_year_counts_by_method(cat, summary=summary)
    years = d.index.values
    perc = get_method_percents(cat, summary=summary)
    order = list(perc.keys())
    labels = {m:f"{m} ({p:.1f}\%)" for m,p in perc.items()}

    for i, y in enumerate(years):
        label={} if i<len(d)-1 else labels
        add_year_data_to_bar_plt(ax, y, order, d, label)

    ax.set_xlim(1987, 2023)
    ax.set_ylim(0.5, 600)
    ax.set_yscale('log')
    plt.legend()
    plt.ylabel("Number of Planets")
    plt.xlabel("Year")
    plt.yscale("log") 
    plt.legend(frameon=False, fontsize=15)
    plt.xlim(min(years), max(years))
    plt.ylim(0.5, 600)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_ticks([1,10,100, 500])
    plt.tick_params(top= False)
    plt.tight_layout()
    plt.savefig("confirmed_planets_vs_time.png", dpi=300)

    
#####

#####


def plot_radii_hist(ax):
    c = load_catalog()
    rcParams["xtick.top"] = False
    ss_radii = load_planet_data()['radius'].to_dict()
    ss_radii = {k:ss_radii[k] for k in ["Earth", "Neptune", "Jupiter"]}
    ss_radii_inv = {v:k for k,v in ss_radii.items()}

    c = c[c["pl_orbper"]<100]
    bins = np.geomspace(0.4, 30, 60)
    cts, b, _ = ax.hist(c.pl_rade, bins=bins, density=False, histtype="step", lw=3, color='gray')

    ax.set_xscale('log')

    ax.tick_params(top= False)
    ax.set_xlabel("Planet Radius [$R_{\oplus}$]")
    ax.set_ylabel("Density")


    tick_bot, ax_top = max(cts)+10, max(cts)+20
    fs = 14
    for n, r in ss_radii.items():
        ax.annotate(n, xy=(r+0.05, tick_bot-30), ha='center', va='bottom', xycoords='data',rotation=20, fontsize=fs)
    ax.vlines(list(ss_radii.values()), ymin=tick_bot, ymax=ax_top, color='k')
    ax.set_ylim(0, ax_top)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    yr = [0, 600]
    
    kw = dict( alpha=0.15, lw=0,zorder=-10,)
    ka = dict(ha='right', va='bottom', xycoords='data', fontsize=fs, rotation=90)
    ytxt = 20
    ECol, NCol, JCol = PLANET_COLORS["Earth"], PLANET_COLORS["Neptune"], PLANET_COLORS['Jupiter']
    ax.fill_betweenx(yr, 1, 1.75, color=ECol,  **kw)
    ax.annotate("Super-Earths", xy=(1.75, ytxt), **ka, color=ECol)
    ax.fill_betweenx(yr, 1.75, 3, color=NCol,  **kw)
    ax.annotate("Sub-Neptune", xy=(3, ytxt), **ka, color=NCol,)
    ax.fill_betweenx(yr, 6, 25, color=JCol,   **kw)
    ax.annotate("Gas Giants", xy=(25, ytxt), **ka, color=JCol)    
    ax.set_xlim(0.4, 30)
    ax.set_yticks([])



def fit_data_range_to_plot(d, p, edge, ax):
        d = d.copy()
        d = d[d["pl_bmasse"] > edge[0]]
        d = d[d["pl_bmasse"] <= edge[1]]

        xData = d.pl_bmasse.values
        yData = d.pl_rade.values

        newx, newy = fit_power(xData, yData, p, edge)
        add_glow(ax, newx, newy)

def fit_power(xData, yData, p, edges):
    fn = lambda x, N: N * (x**p)
    popt, pcov = curve_fit(fn, xData, yData)
    xnew = np.linspace(edges[0], edges[1], 5000)
    return xnew, fn(xnew, *popt)


def add_glow(ax, x, y, n_lines=10,diff_linewidth=1.05, alpha_value=0.03, color="navy",):
    for n in range(1, n_lines+1):
        kwgs = dict(
            linewidth=2+(diff_linewidth*n),
            alpha=alpha_value,
            color=color,
        )
        ax.plot(x, y, **kwgs)
    
def plot_radius_mass_relation_plot(ax):
    MJUP = 317.8
    MSUN = 333000

    EDGE = [1e-4, 2, 0.41*MJUP, 0.08*MSUN] #, 10**6]
    POWERS = [0.28, 0.59, -0.04] #  0.88]

    ax.set_xlim(0.01, 5000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Planet Mass [$M_{\oplus}$]")
    ax.set_ylabel("Planet Radius [$R_{\oplus}$]")


    yr = [0.001, 300]
    ax.vlines(EDGE, ymin=yr[0], ymax=yr[1], linewidth = 0.5, linestyle= "--", color="gray")    

    cat = load_catalog()
    c = squash_small_numeber_types(cat)
    c = c[["pl_bmasse", "pl_rade"]].dropna()


    x, y = c.pl_bmasse, c.pl_rade
    x_space = np.logspace(np.log10(min(x)), np.log10(max(x)), 50)
    y_space = np.logspace(np.log10(min(y)), np.log10(max(y)), 50)


    ax.scatter(x, y, s=0.5, label="Exoplanets", alpha=0.3, zorder=0, color='k')
    ax.set_xlim(min(c.pl_bmasse), EDGE[-1])
    ax.set_ylim(min(c.pl_rade), max(c.pl_rade))
    ax.tick_params('x', top=True)

    for i, p in enumerate(POWERS): 
        fit_data_range_to_plot(c, p, [EDGE[i], EDGE[i+1]] , ax)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:,g}'.format(y)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    

    # add dash line labels
    fs = 14
    yl = 0.4
    kw = dict(ha='right', va='bottom', xycoords='data',rotation=90, fontsize=fs, color='gray')
    ax.annotate("$2 M_{\oplus}$", xy=(EDGE[1], yl), **kw)
    ax.annotate("$0.41 M_{J}$", xy=(EDGE[2], yl), **kw)

    kw = dict(ha='center', va='bottom', xycoords='data', fontsize=fs, color='gray')
    ax.annotate("volatile\n envelope", xy=(EDGE[1], 30), **kw)
    ax.annotate("self\n compression", xy=(EDGE[2], 30), **kw)
    
    kw = dict(ha='center', va='bottom', xycoords='data', fontsize=fs, color='navy', alpha=1)
    ax.annotate("$R\\sim M^{0.28}$\nRocky\nPlanets", xy=(0.25, 1.5), **kw)
    ax.annotate("$R\\sim M^{0.59}$\nNeptunian\nPlanets", xy=(14, 0.35), **kw)
    ax.annotate("$R\\sim M^{-0.04}$\nJovian\nPlanets", xy=(2000, 1), **kw)
    pdata = load_planet_data().T.to_dict()
    for p in ["Earth", "Mercury", "Jupiter", "Neptune"]:
        pM,pR = pdata[p]['mass'], pdata[p]['radius']
        ax.scatter(pM,pR,color=PLANET_COLORS[p], s=50, marker="o",zorder=100)
    

        
    

def make_radii_and_mass_plots():
    fig, ax = plt.subplots(1,2, figsize=(13,5))    
    plot_radius_mass_relation_plot(ax[0])
    plot_radii_hist(ax[1])
    add_subplot_letter(ax)
    plt.tight_layout()
    plt.savefig("radii_and_mass_relations.png")


#####

####



def mass_to_lumin_for_main_sequence(m):
    """m (Msun) to L (Lsun)
    https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
    """
    if m < 0.43:
        return 0.23 * np.power(m, 2.3)
    elif m < 2:
        return np.power(m, 4)
    elif m < 55:
        return 1.4 * np.power(m, 3.5)
    else:
        return 32000 * m
    
    

def get_luminosity_ratio(star_radius_km, star_temp_k):
    '''
    Given the radius and temperature of a star on the main sequence,
    estimate & return the luminosity ratio (L-star / L-sun).
    '''
    return np.power(arg_star_km / R_SUN, 2) \
            * np.power(arg_star_temp / T_SUN, 4)

def get_hz(m_star=None, l_star=None, t_star=None, r_star=None):
    '''
    https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html
    Given the luminosity of a star on the main sequence,
    estimate & return its habitable zone boundaries in AUs.
    '''
    if m_star:
        l_star = mass_to_lumin_for_main_sequence(m_star)
    if l_star:
        lum_ratio = l_star  
    else:
        lum_ratio = get_luminosity_ratio(r_star, t_star)
    sqrtratio = np.sqrt(lum_ratio)
    _r_inner = 0.75 * sqrtratio # innermost boundary
    _r_center = 1.0 * sqrtratio # innermost boundary
    _r_outer = 1.77 * sqrtratio # outermost boundary
    return np.array([_r_inner, _r_center, _r_outer]) 



def smooth_xydata(x,y, smooth=0.9):
    return x, gaussian_filter1d(y, smooth)
    



def check_if_inside(data_x, data_y, x1, y1, x2, y2):
    c1 = data_x > np.interp(data_y, y1,x1)
    c2 = data_x < np.interp(data_y, y2,x2)
    return c1&c2



def plot_habitable_zone(ax):
    cat = load_catalog()
    inner = pd.read_csv("data/hz_inner.csv").sort_values(by='mass')
    outer = pd.read_csv("data/hz_outer.csv")



    inner_x, inner_y = smooth_xydata(inner.mass, inner.dist, 0.23)
    outer_x, outer_y = smooth_xydata(outer.mass, outer.dist, 1.2)

    c = "mediumaquamarine"
    ax.fill(
        np.append(inner_x, outer_x[::-1]),
        np.append(inner_y, outer_y[::-1]),
        color=c, alpha=0.75, label="Habitable Zone"
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    # glow effect
    n_lines = 20
    diff_linewidth = 1.05
    alpha_value = 0.03
    for n in range(1, n_lines+1):
        kwgs = dict(
            linewidth=2+(diff_linewidth*n),
            alpha=alpha_value,
            color=c
        )
        ax.plot(inner_x, inner_y, **kwgs)
        ax.plot(outer_x, outer_y , **kwgs)


    ss_stats = load_planet_data()
    ss_distances = ss_stats.to_dict()['distance']
    ss_distances = {k:ss_distances[k] for k in ["Mercury", "Earth", "Neptune", "Jupiter"]}

    cat_x, cat_y = cat.pl_orbsmax.values, cat.st_mass.values
    inside = check_if_inside(cat_x, cat_y, inner_x, inner_y, outer_x, outer_y)
    in_x, in_y = cat_x[inside], cat_y[inside]

    ax.scatter(cat_x, cat_y, alpha=0.25, s=0.1, color='k')
    ax.scatter(in_x, in_y, s=10, alpha=1., color='k', zorder=10, marker=".", label="Exoplanets")

    ax.set_xlim(0.01, 10)
    ax.set_ylim(min(inner_y), max(outer_y))
    ax.set_xlabel("Distance from Star [Au]")
    ax.set_ylabel("Star Mass [$M_{\odot}$]")
    ax.tick_params(axis='x', pad=10, top=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_yticks([0.2,0.5,1])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for i, (n, d) in enumerate(ss_distances.items()):
        ax.scatter(d, 1, s=50, label=n, c=PLANET_COLORS[n], zorder=20, marker="o")

    ax.legend(frameon=False, fontsize=14, markerscale=2)

    




def read_planet_category_ranges():
    d = pd.read_csv("data/planet_categories.csv")
    data = {}
    for c in d.category.unique():
        df = d[d["category"]==c][["period", "mass"]]
        pts = df.values.tolist() # top_left, bot_right
        width = np.abs(pts[0][0] - pts[1][0])
        height = np.abs(pts[0][1] - pts[1][1])
        pt_bottom_left = (pts[0][0], pts[0][1]-height)
        data[c] =  (pt_bottom_left, width, height)
    return data


def add_rectangles(ax, data, colors):
    for d, c in zip(data.values(), colors):
        pc = PatchCollection(
            [Rectangle(*d)], facecolor="none", alpha=1, edgecolor=c, lw=2)
        ax.add_collection(pc)


        
def plot_mass_period_diagram(ax):
    ZONES = {
        "Hot Jupiters": [(0.33, 27000), "salmon"],
        "USP Planets": [(0.2, 0.0185), "khaki"],
        "Classic Giants": [(2200, 27000), "teal"],
        "Warm Neptunes": [(500, 20), "lightskyblue"],
        "Super Earths":[(100, 2), "mediumseagreen"],
        "Rocky Planets":[(36 ,0.25), "sandybrown"]
    }    

    c = load_catalog()
    ax.scatter(c.pl_orbper, c.pl_bmasse, s=0.5, color="k")
    planet_corners = read_planet_category_ranges()
    planet_colors = [ZONES[k][-1] for k in planet_corners.keys()]
    add_rectangles(ax, planet_corners, planet_colors)
    
    ss_data = load_planet_data().T.to_dict()
    for k in ["Jupiter", "Neptune", "Earth", "Mercury"]:
        pT,pM = ss_data[k]['period'],ss_data[k]['mass']
        ax.scatter(pT, pM, color=PLANET_COLORS[k], marker="o", s=50 )

    fs = 14
    kw = dict(
        ha='left', va='bottom', xycoords='data', fontsize=fs, color='black', fontweight='extra bold'
    )
    for n, d in ZONES.items():
        bbox=dict(fc=d[1], lw=0)
        ax.annotate(n, xy=d[0], **kw, bbox=bbox)
        
    ax.set_xscale('log')
    ax.set_yscale('log')    
    ax.set_xlim(0.1, 11**5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:,g}'.format(y)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:,g}'.format(y) if y < 1000 else '$10^{' + str(int(np.log10(y))) + '}$'))
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Mass [$M_{\oplus}$]")

    
def make_categories_plot():
    set_matplotlib_style_settings(major=7, minor=4, linewidth=1.5, grid=False, topon=True, righton=True)
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    plot_mass_period_diagram(ax[0])
    plot_habitable_zone(ax[1])
    add_subplot_letter(ax, inside=False)
    plt.tight_layout()
    plt.savefig("scatter_categories.png", dpi=300)

    

# -

# # Plots

plot_counts_per_year(True)
make_categories_plot()
make_radii_and_mass_plots()

# # Notes
#
# ### Planet Radius vs Mass
#
# ![](https://astrobites.org/wp-content/uploads/2013/12/comp.png)
#
# https://www.science.org/doi/10.1126/science.aah4097
#
# ![](https://planetary.s3.amazonaws.com/web/assets/pictures/_2400x1799_crop_center-center_82_line/11550/20160411_exoplanet-mass-radius.jpg.webp)
#
# ### Radius gap
#
# ![](https://static.wixstatic.com/media/b5d8a1_ce5716bfe945405f9fb06363e4ff6a35~mv2.png/v1/fill/w_1766,h_1184,al_c,q_90/b5d8a1_ce5716bfe945405f9fb06363e4ff6a35~mv2.webp)
#
# ![](http://4.bp.blogspot.com/-1_JwSFO2IRM/Ulrs11-P6RI/AAAAAAAAASg/CUmCbCdprgc/s1600/Radii+of+Transiting+Planets+-+July+2013.gif)
#
# ![](https://www.researchgate.net/profile/Eric-Wolf/publication/337184533/figure/fig1/AS:824257319821313@1573529667328/Classification-of-exoplanets-into-different-categories-Kopparapu-et-al-2018-The.ppm)
#
# ### Exoplanet radius vs distance from the star (Hot neptune destert)
#
# ![](https://cdn.spacetelescope.org/archives/images/screen/opo1852b.jpg)
#
#
# ### Exoplanet discovery space (radius and mass VS period)
#
# ![](https://www.pnas.org/cms/10.1073/pnas.1304196111/asset/96d2f5b2-0446-4bc9-994a-34451261a73c/assets/graphic/pnas.1304196111fig01.jpeg)
#
# ![](https://ars.els-cdn.com/content/image/1-s2.0-S027510622030062X-gr1.jpg)
#
# ### Habitable Zone
#
# https://arxiv.org/pdf/1301.6674.pdf
