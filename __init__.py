import openpifpaf

from .src import wholebodykp

def register():
    openpifpaf.DATAMODULES['wholebodykp'] = wholebodykp.WholeBodyKp
