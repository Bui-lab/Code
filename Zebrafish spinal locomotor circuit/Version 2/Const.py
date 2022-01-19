import matplotlib.cm as colormap
from matplotlib import pyplot as plt, rcParams

#https://matplotlib.org/stable/tutorials/colors/colormaps.html
#Declare color maps
__cmapGrey = colormap.get_cmap('Greys')
__cmapBlue = colormap.get_cmap('Blues')
__cmapRed = colormap.get_cmap('Reds')
__cmapGreen = colormap.get_cmap('Greens')
__cmapBrown = colormap.get_cmap('YlOrBr')
__cmapRedPurple = colormap.get_cmap('RdPu')
__cmapBrightGreen = colormap.get_cmap('hsv')
__cmapYellowGreen = colormap.get_cmap('YlGn')
__cmapYellow = colormap.get_cmap('Oranges')
__cmapPurple = colormap.get_cmap('Purples')

IPSI_COLORS = {
    'IC': __cmapBrown(0.9), 
    'dI6': __cmapBrightGreen(0.5),
    'MN': __cmapRed(0.6), 
    'V0d': __cmapGreen(0.6), 
    'V0v': __cmapBlue(0.6), 
    'V1': __cmapYellow(0.6), 
    'V2a': __cmapRedPurple(0.5),
    'Muscle': __cmapPurple(0.5),
    'Left': __cmapBlue(0.5)
}
CONTRA_COLORS = {
    'IC': 'Grey', 
    'dI6': 'Grey',
    'MN': 'Grey', 
    'V0d': 'Grey', 
    'V0v': 'Grey', 
    'V1': 'Grey',
    'V2a': 'Grey',
    'Muscle': 'Grey',
    'Right': __cmapRed(0.5)}
    
IPSI_COLOR_MAPS = {
    'IC': __cmapBrown, 
    'dI6': __cmapYellowGreen, #different color than individual dI6 color
    'MN': __cmapRed, 
    'V0d': __cmapGreen, 
    'V0v': __cmapBlue, 
    'V1': __cmapYellow, 
    'V2a': __cmapRedPurple,
    'Muscle': __cmapPurple,
    'Left': __cmapBlue
}
CONTRA_COLOR_MAPS = {
    'IC': __cmapGrey, 
    'dI6': __cmapGrey,
    'MN': __cmapGrey, 
    'V0d': __cmapGrey, 
    'V0v': __cmapGrey, 
    'V1': __cmapGrey,
    'V2a': __cmapGrey,
    'Muscle': __cmapGrey,
    'Right': __cmapRed}

#declare line length and width
MULTIPANEL_LINELENGTH = 0.5
MULTIPANEL_LINEWIDTH = 1

MULTIPANEL_SMALLER_SIZE = 12
MULTIPANEL_SMALL_SIZE = 12
MULTIPANEL_FONT_STYLE = 'normal'

#declare y-axis limits
MULTIPANEL_LOWER_Y = -70
MULTIPANEL_UPPER_Y = 20
