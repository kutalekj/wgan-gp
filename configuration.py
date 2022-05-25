"""User guided car image colorization with WGAN-GP.

Authors
-------
Chimenti Andrea     xchime00
Chudarek Ales       xchuda04
Kosak Vaclav        xkosak01
Kutalek Jiri        xkutal09

Insitution
----------
Brno University of Technology
Faculty of Information Technology

Date
----
May 2022

"""

from configparser import ConfigParser


def config(filename="config.ini", section=""):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    data = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            data[param[0]] = param[1]
    else:
        raise Exception("Section {0} not found in the {1} file".format(section, filename))

    return data
