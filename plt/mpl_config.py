import matplotlib.pyplot as plt

def set_style():
    """
    Standardizes Matplotlib font and style settings for the project.
    Returns the project-specific color palette.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
        "font.size": 18
    })
    
    color = ['#F94141', '#589FF3', '#37AB78', '#F3B169', '#808080']
    return color
