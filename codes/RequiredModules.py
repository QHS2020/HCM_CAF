# -*- coding: utf-8 -*-

import sympy
import control
import filterpy
from filterpy.kalman import ExtendedKalmanFilter as EKF
#from datetime import datetime,timedelta
#import json

import networkx as nx
#from graphviz import Digraph as Digraphviz

from intersect import intersection as IntersectionCurves

import sdeint

import builtins
import itertools
import pickle

import time

from intersect import intersection

import os
import sys
from imp import reload
import copy
import argparse
import shapely
#from shapely.geometry import Point
#from shapely.geometry import LineString
#from shapely.ops import split as shapely_split
#from shapely.geometry import Polygon
#from shapely.ops import nearest_points
from filterpy.stats import plot_covariance_ellipse

import glob
import pyinter
#sparse matrix
import scipy.sparse as ScipySparse
import scipy


import random
import numpy as np
import math

#
import plotly.graph_objects as plotly_go



from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

import pandas as pd

import warnings
#warnings.filterwarnings("error")

#from graphviz import Digraph

##matplotlib
import matplotlib.pyplot as plt
#from matplotlib.patches import Circle, Wedge


#import intervaltree
#import pyinter#python interval class


import shapely
from shapely.geometry import Polygon
from shapely.geometry.multipolygon import MultiPolygon

#union multiplogyon to single polygon
from shapely.ops import unary_union


#from bisect import bisect_left

#used for interp to get the points


def my_pickle_open(filename, readmode="r"):
        
        return pickle.load( open( filename, readmode ) )



def my_pd_to_rst_table(data_pd, filename = 'temp.txt'):
        """
        print a pandas table to restructuredtext and write it to the file given by arg filename. 
        
        """
        
        f = open(filename,'w')
        from tabulate import tabulate
        f.write(tabulate(data_pd, tablefmt="rst", headers = data_pd.columns))
        f.close()

def my_json_load(filename):
        """
        
        """
        
        
        pass

