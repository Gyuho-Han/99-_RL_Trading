import os
LOGGER_NAME = 'rltrader'
BASE_DIR = os.environ.get('RLTRADER_BASE',
    os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))
