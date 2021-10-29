import dash
from dash import html, dcc, Input, Output

from App import LayoutMain
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
    app.run_server(debug=True)

