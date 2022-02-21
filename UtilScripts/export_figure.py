import os
from pathlib import Path

export_path = os.environ['EXPPATH']


# exports the figure as html with the name to
# the path given by EXPPATH and adds the fitting number if the file already exists
def export_plot_as_html(figure, name):
    path = export_path + name + '.html'
    counter = 0
    # make sure to not overwrite a file
    while Path(path).is_file():
        path = export_path + name + '_' + str(counter) + '.html'
        counter = counter + 1

    figure.write_html(path)

