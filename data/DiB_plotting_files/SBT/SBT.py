from shapely.geometry import Point, Polygon
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from collections import defaultdict

# TODO: Implement for loop to optimize code

"""
Reads the co-ordinates of the Curves saved as csv files.
"""

directory = Path(__file__).parent / 'sbt_chart'
# path_to_chart = directory / f'Curve{i}.csv'

dict_of_dataframes = defaultdict()
dict_of_curves = defaultdict()

for i in range(1, 31):  # Hardcoded 28, this is the number of curves
    dict_of_dataframes[f'df_C{i}'] = pd.read_csv(directory / f'Curve{i}.csv', sep=';')

# TODO: change the numbering of the files to something more reader-friendly
df_C4_P5 = pd.read_csv(directory / "Curve4_P5.csv", sep=';')
df_C10_P4 = pd.read_csv(directory / "Curve10_P4.csv", sep=';')
df_C26_P8 = pd.read_csv(directory / "Curve26_P8.csv", sep=';')

"""
Converts the dataframes to list
"""

for i in range(1, 31):
    print(dict_of_curves)
    dict_of_curves[f'C{i}'] = dict_of_dataframes[f'df_C{i}'].values.tolist()

# TODO: change the numbering of the files to something more reader-friendly
# C4_P5 = df_C4_P5.values.tolist()
# C10_P4 = df_C10_P4.values.tolist()
# C26_P8 = df_C26_P8.values.tolist()
dict_of_curves['C4_P5'] = df_C4_P5.values.tolist()
dict_of_curves['C10_P4'] = df_C10_P4.values.tolist()
dict_of_curves['C26_P8'] = df_C26_P8.values.tolist()
"""
Finding the co-ordinates of Polygons i.e. the soil zones
"""

P1 = Polygon(
    dict_of_curves['C1'] + dict_of_curves['C2'] + dict_of_curves['C3'] + dict_of_curves['C4'] + dict_of_curves['C5'])
P2 = Polygon(dict_of_curves['C6'] + dict_of_curves['C7'] + dict_of_curves['C8'])
P3 = Polygon(
    dict_of_curves['C8'] + dict_of_curves['C2'] + dict_of_curves['C10'] + dict_of_curves['C12'] + dict_of_curves['C11'])
P4 = Polygon(dict_of_curves['C14'] + dict_of_curves['C10_P4'] + dict_of_curves['C3'] + dict_of_curves['C13'])
P5 = Polygon(dict_of_curves['C13'] + dict_of_curves['C25'] + dict_of_curves['C24'] + dict_of_curves['C23']
             + dict_of_curves['C4_P5'])
P6 = Polygon(dict_of_curves['C24'] + dict_of_curves['C22'] + dict_of_curves['C27'] + dict_of_curves['C19']
             + dict_of_curves['C26'])
P7 = Polygon(dict_of_curves['C20'] + dict_of_curves['C21'] + dict_of_curves['C27'])
P8 = Polygon(dict_of_curves['C25'] + dict_of_curves['C26_P8'] + dict_of_curves['C18'] + dict_of_curves['C15'])
P9 = Polygon(
    dict_of_curves['C15'] + dict_of_curves['C14'] + dict_of_curves['C12'] + dict_of_curves['C16'] + dict_of_curves[
        'C17'])
P10 = Polygon(dict_of_curves['C28'] + dict_of_curves['C29'] + dict_of_curves['C30'])


def find_sbt(coordinates):
    """
    Function to find SBT based on Non-normalized CPT Soil Behavior Type (SBT) chart (Robertson et al., 1986,
    updated by Robertson, 2010).
    # Note: This function does not return the right value if the point lies in the border of the soil zones.
    out of the graph
    """

    if coordinates.within(P1):
        sbt = "Sensitive, fine grained"
        sbtn = 1.0
    elif coordinates.within(P2):
        sbt = "Organic soils - clay"
        sbtn = 2.0
    elif coordinates.within(P3):
        sbt = "Clay – silty clay to clay"
        sbtn = 3.0
    elif coordinates.within(P4):
        sbt = "Silt mixtures – clayey silt to silty clay"
        sbtn = 4.0
    elif coordinates.within(P5):
        sbt = "Sand mixtures – silty sand to sandy silt"
        sbtn = 5.0
    elif coordinates.within(P6):
        sbt = "Sands – clean sand to silty sand"
        sbtn = 6.0
    elif coordinates.within(P7):
        sbt = "Gravelly sand to dense sand"
        sbtn = 7.0
    elif coordinates.within(P8):
        sbt = "Very stiff sand to clayey sand"
        sbtn = 8.0
    elif coordinates.within(P9):
        sbt = "Very stiff fine grained"
        sbtn = 9.0
    elif coordinates.within(P10):
        sbt = "Organic soils - clay"
        sbtn = 2.0
    else:
        sbt = "Point lies in the border"
        sbtn = 0.0
    result = [sbt, sbtn]
    return result[1]


def find_sbt_Robertson_2016(x, y):
    """
    Equations of the curves in Robertson 2016 plot
    Curve CD = 70:(((y - 11)*((1 + 0.06 * x)**17)-70)
    Curve IB = 32:(100*(y+10)/(32*(70+x*y))-1)
    Curve IB = 22:(100*(y+10)/(22*(70+x*y))-1)
    """
    CD70 = ((y - 11) * ((1 + 0.06 * x) ** 17) - 70)
    IB32 = (100 * (y + 10) / (32 * (70 + x * y)) - 1)
    IB22 = (100 * (y + 10) / (22 * (70 + x * y)) - 1)
    print('CD70 is', CD70)
    print('IB32 is', IB32)
    print('IB22 is', IB22)

    if (CD70 < 0) and (IB32 > 0):
        print('SC')
        sbtn = 6.0
    elif (CD70 < 0) and (IB32 < 0) and (IB22 > 0) and (y > 10):
        print('TC')
        sbtn = 4.0
    elif (0.1 < x < 2.066) and (1 < y < 10):
        print('CCS')
        sbtn = 1.0
    elif IB22 < 0 and CD70 < 0 and 1 < y < 10 and 2 < x < 10:
        print('CC')
        sbtn = 2.0
    elif IB22 < 0 and CD70 > 0:
        print('CD')
        sbtn = 3.0
    elif IB22 > 0 and IB32 < 0 and CD70 > 0:
        print('TD')
        sbtn = 5.0
    elif IB32 > 0 and CD70 > 0:
        print('SD')
        sbtn = 7.0
    else:
        print('Point lies outside')
    return sbtn


def robertson2010_plot(pnts: pd.DataFrame() = None, title: str = None, output=None):
    """Plot function to show CPT measurements on the Robertson 2010 chart"""

    # Create axis figure and title
    fig = plt.figure(figsize=(10, 8))
    if title is None:
        title = 'Robertson 2010 SBT chart (Rf - qt/pa)'
    fig.suptitle(f'{title}', fontsize=16)
    ax = fig.add_subplot(111)

    # Add polygons to list
    pols = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]

    # Loop over polygons to plot them and annotate a number
    for nr, pol in enumerate(pols):
        ax.plot(*pol.exterior.xy, c='k')
        x, y = pol.centroid.x, pol.centroid.y
        ax.annotate(nr + 1, (x, y))

    # Define limits
    ax.set_xlim(0.1, 20)
    ax.set_ylim(1, 1000)

    # Define x and y labels and set to log scale
    ax.set_ylabel('qt/pa')
    ax.set_xlabel('Rf')
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Check if there are any pnts available, if so plot them.
    if pnts is not None:
        if not isinstance(pnts, pd.DataFrame):  # Only pd.DataFrame is excepted
            NotImplementedError('Only the use of type pandas Dataframe is implemented for pnts.')
        # If a z column (defining the depth) is present, then use this column to change the color of the pnts
        if 'z' in pnts.columns:
            ax.scatter(pnts['Rf'], pnts['qt'] / 0.1, c=pnts['z'])
        # Otherwise all pnts have the same color
        else:
            ax.scatter(pnts['Rf'], pnts['qt'] / 0.1)

    if output:
        plt.savefig(f'{os.path.join(output, title)}.png')
        plt.close()
    else:
        plt.show()

def get_SBT(rf, qt, classf_method='Robertson2010'):
    """Determination of SBT based on qt and rf
    :param rf: all Rf values within a cpt
    :type rf: Union[np.ndarray or list]
    :param qt: all qt values within a cpt
    :type qt: Union[np.ndarray or list]
    :return: SBT type (which is a integer) for all combinations of rf/qt
    :rtype: list
    """

    if classf_method == 'Robertson2010':
        if len(rf) != len(qt):
            raise ValueError(f'length of rf; {len(rf)} and qt; {len(qt)} are not equal')
        return [find_sbt(Point(a, b / 0.1)) for a, b in zip(rf, qt)]
    elif classf_method == 'Robertson2016':
        if len(rf) != len(qt):
            raise ValueError(f'length of rf; {len(rf)} and qt; {len(qt)} are not equal')
        return [find_sbt_Robertson_2016(a, b / 0.1) for a, b in zip(rf, qt)]
    elif classf_method == 'Robertson2009':
        if len(rf) != len(qt):
            raise ValueError(f'length of Fr; {len(rf)} and qt; {len(qt)} are not equal')
        return [find_sbt(Point(a, b)) for a, b in zip(rf, qt)]
    else:
        NotImplementedError(f'Classification method "{classf_method}" is not implemented')
        
        
    return

# robertson2010_plot()
