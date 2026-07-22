import pathlib

# ============================================== #
# DATABASE
# ============================================== #

database = dict(name = 'iam-diagnostics-internal',
                username = 'none',
                password = 'none')

figures = pathlib.Path(__file__).parent.parent.parent / 'Figures'
