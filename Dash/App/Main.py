import dash
from dash import html

from Dash.App import LayoutMain
from DataFormats.DbInstance import DbInstance

# ----------------------------------------------------------------------------------------------------------------------
# Handles the creation of the actual Html Website and notifies the Checks for layout Changes
# ----------------------------------------------------------------------------------------------------------------------

# create app and read in the files
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

db_instance = DbInstance()

# init the first layout
app.layout = html.Div(LayoutMain.init_layout())
LayoutMain.register_callbacks(app, db_instance)

# ----------------------------------------------------------------------------------------------------------------------

# starts the application
if __name__ == '__main__':
    # use_reloader needs to be disabled when using the pycharm debugger.
    # if hot reloading is needed in normal execution it can be set to True
    app.run_server(debug=True, use_reloader=False)

