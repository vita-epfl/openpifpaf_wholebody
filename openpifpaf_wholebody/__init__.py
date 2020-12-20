import openpifpaf

from . import wholebodykp

def register():
    openpifpaf.DATAMODULES['wholebodykp'] = wholebodykp.WholeBodyKp