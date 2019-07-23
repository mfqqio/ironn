import numpy as np
import pytest
import unittest

import sys
sys.path.append("../ironn/modules/preprocessing")
from convexhull import MyJarvisWalk

#@pytest.fixture(scope="class")


test_class = MyJarvisWalk()

class TestConvex(unittest.TestCase):
    def test_compare(self, a = 10, b = 6):
        """
        Tests comparison between 'a' and 'b'.
        Parameters:
        ----------
            a (str/flt): String or float 'a' to be compared
            b (str/flt): String or float 'b' to be compared
        Returns:
        ----------
            -1, 0, 1 if a < b, a = b, or a > b accordingly.
            or Assertion Error
        """
        assert((a > b) - (a < b)) in (-1,0,1)
      
    def test_turn(self, p = [5, 3, 4], q = [9, 3, 4], r = [13, 3, 4]):
        """
        Turns right, straight or left.
        Parameters:
        ----------
            p(array): Point where we start
            q(array): Possible point to go
            r(array): Possible point to go
        Returns:
        ----------
            -1, 0, 1 if p, q, r forms a right, straight, or left turn.
        """
        assert test_class.compare((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0) in (-1,0,1), "Coordinates must be wrong!"
        

    def test_dist(self, p = (5, 3), q = (2, 4)):
        """
        Returns the squared Euclidean distance between p and q.
        Parameters:
        ----------
            p(array): point 'p'
            q(array): point 'q'
        Returns:
        ----------
            test_dist(int) : Squared Euclidean distance
        """
        assert(test_class._dist(p, q) == 10)
    
    def test_next_hull_pt(self, points = [(2,3), (4,2), (3,2)], p = (0,0)):
        """
        Returns the next point on the convex hull from p.
        Parameters:
        ----------
            points: list of tuples of points (x, y).
            p(tuple): list of starting point
        Returns:
        ----------
            q(tuple): next point in the Convex-Hull in the points array
        """
        assert test_class._next_hull_pt(points, p) == (4,2)

    def test_convex_hull(self, points = [(0,0),(2,3), (4,2), (3,2)]):
        """
        Returns the points on the convex hull in Counterclockwise order.
        Parameters:
        ----------
            points(array): list of tuples of points (x, y).
        Example:
        ----------
            convex_hull(flat_coords)
        """
        assert test_class.convex_hull(points) == [(0,0), (4,2), (2,3), (0,0)]