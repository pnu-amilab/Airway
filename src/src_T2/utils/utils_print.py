import logging


def print_log(text='', end='\n'):
    print(text, end=end)
    logging.info(text)
