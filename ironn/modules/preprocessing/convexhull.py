class MyJarvisWalk():

    def compare(self, a, b):
        """
        Compares 'a' and 'b'.

        Parameters:
        ----------
        a: str/flt
            String or float 'a' to be compared
        b: str/flt
            String or float 'b' to be compared

        Returns:
        ----------
        -1, 0, 1 if a < b, a = b, or a > b accordingly.
        """
        return (a > b) - (a < b)

    def turn(self, p, q, r):
        """
        Turns right, straight or left.

        Parameters:
        ----------
        p: array
            Point where we start
        q: array
            Possible point to go
        r: array
            Possible point to go

        Returns:
        ----------
        -1, 0, 1 if p, q, r forms a right, straight, or left turn.
        """
        return self.compare((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _dist(self, p, q):
        """
        Returns the squared Euclidean distance between p and q.

        Parameters:
        ----------
        p: array
            point 'p'
        q: array
            point 'q'

        Returns:
        ----------
        Squared Euclidean distance
        """
        dx, dy = q[0] - p[0], q[1] - p[1]
        return dx * dx + dy * dy

    def _next_hull_pt(self, points, p):
        """
        Returns the next point on the convex hull from p.

        Parameters:
        ----------
        points: list
            list of tuples of points (x, y).
        p: tuple
            list of starting point

        Returns:
        ----------
        q: tuple
            next point in the Convex-Hull in the points array
        """
        TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)
        q = p
        for r in points:
            t = self.turn(p, q, r)
            if t == TURN_RIGHT or t == TURN_NONE and self._dist(p, r) > self._dist(p, q):
                q = r
        return q

    def convex_hull(self, points):
        """
        Returns the points on the convex hull in Counterclockwise order.

        Parameters:
        ----------
        points: list
            list of tuples of points (x, y).

        Returns:
        --------
        hull: list
            list of tuples of points that represent coordinates of convex hull
        """
        #hull = points.sort(key=lambda x: x[0])
        hull = [min(points)]
        for p in hull:
            q = self._next_hull_pt(points, p)
            if q != hull[0]:
                hull.append(q)
        hull.append(hull[0])
        return hull
