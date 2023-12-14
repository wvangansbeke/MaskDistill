#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os


class Path(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/esat/rat/wvangans/Datasets/'
        db_names = ['VOCSegmentation', 'VOCDetection']

        if database == '':
            return db_root

        if database == 'VOCSegmentation':
            return os.path.join(db_root, database)

        elif database == 'VOCDetection':
            return os.path.join(db_root, 'VOCdevkit/VOC2012')

        else:
            raise ValueError('Invalid database {}'.format(database))    
