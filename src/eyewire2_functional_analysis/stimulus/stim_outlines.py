import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# -------------------------------------------------------------------------------
def movingBar(
        lEdge: float = 300, trajLen: float = 4000,
        angles: list = [0, 45, 90, 135],
        x0: float = 0,
        y0: float = 0,
        FOV_diam: float | None = None
    ):
    """Create a Shapely polygon representing the union of moving-bar outlines.

    Builds one rectangle per entry in ``angles`` (bar width = ``lEdge``,
    length = ``trajLen``) centred at ``(x0, y0)``, then returns their union.
    Optionally intersects the union with a circular FOV aperture.

    Args:
        lEdge: Bar edge length (width) in stimulus units.
        trajLen: Bar trajectory length in stimulus units.
        angles: List of bar-motion angles in degrees.
        x0: X coordinate of the centre point.
        y0: Y coordinate of the centre point.
        FOV_diam: Diameter of the circular field-of-view aperture to intersect
            with. If ``None``, no aperture is applied.

    Returns:
        shapely.geometry.base.BaseGeometry: Union polygon of all bar outlines,
        optionally clipped to the FOV aperture.
    """

    def _create_rotated_rect(w, h, a, cx, cy):
        """Return a Shapely Polygon for a rectangle rotated by ``a`` degrees and centred at ``(cx, cy)``.

        Args:
            w: Rectangle width.
            h: Rectangle height.
            a: Rotation angle in degrees (counter-clockwise).
            cx: X coordinate of the centre.
            cy: Y coordinate of the centre.

        Returns:
            shapely.geometry.Polygon: The rotated, translated rectangle.
        """
        # Create rectangle centered at origin
        half_w = w / 2
        half_h = h / 2

        # Define corner points
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])

        # Rotate
        angle_rad = np.radians(a)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated_corners = corners @ rotation_matrix.T

        # Translate to center
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        return Polygon(rotated_corners)


    # Create all rectangles
    rectangles = []
    for angle in angles:
        rect = _create_rotated_rect(lEdge, trajLen, angle, x0, y0)
        rectangles.append(rect)

    # Create union of all rectangles to form the final polygon
    union_polygon = unary_union(rectangles)

    # Apply FOV aperture if specified
    if FOV_diam is not None:
        aperture = Point(x0, y0).buffer(FOV_diam / 2)
        return union_polygon.intersection(aperture)

    return union_polygon

# -------------------------------------------------------------------------------
def spot(
        diam: float = 1000,
        x0: float = 0,
        y0: float = 0,
        FOV_diam: float | None = None
    ):
    """Create a Shapely circular polygon representing a full-field spot stimulus.

    Args:
        diam: Diameter of the spot in stimulus units.
        x0: X coordinate of the centre.
        y0: Y coordinate of the centre.
        FOV_diam: Diameter of the circular FOV aperture to intersect with.
            If ``None``, no aperture is applied.

    Returns:
        shapely.geometry.base.BaseGeometry: Circle polygon, optionally clipped
        to the FOV aperture.
    """
    # Create circle centered at (x0, y0)
    circle_polygon = Point(x0, y0).buffer(diam / 2)

    # Apply FOV aperture if specified
    if FOV_diam is not None:
        aperture = Point(x0, y0).buffer(FOV_diam / 2)
        return circle_polygon.intersection(aperture)

    return circle_polygon

# -------------------------------------------------------------------------------
def box(
        dx: float = 1000,
        dy: float = 1000,
        angle: float = 0,
        x0: float = 0,
        y0: float = 0,
        FOV_diam: float | None = None
    ):
    """Create a Shapely rectangular polygon, optionally rotated and clipped to a FOV aperture.

    Args:
        dx: Rectangle width in stimulus units.
        dy: Rectangle height in stimulus units.
        angle: Rotation angle in degrees (counter-clockwise).
        x0: X coordinate of the centre.
        y0: Y coordinate of the centre.
        FOV_diam: Diameter of the circular FOV aperture to intersect with.
            If ``None``, no aperture is applied.

    Returns:
        shapely.geometry.base.BaseGeometry: Rotated rectangle polygon, optionally
        clipped to the FOV aperture.
    """
    # Create rectangle centered at origin
    half_w = dx / 2
    half_h = dy / 2

    # Define corner points
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])

    # Rotate
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    rotated_corners = corners @ rotation_matrix.T

    # Translate to center
    rotated_corners[:, 0] += x0
    rotated_corners[:, 1] += y0

    # Create polygon
    rect_polygon = Polygon(rotated_corners)

    # Apply FOV aperture if specified
    if FOV_diam is not None:
        aperture = Point(x0, y0).buffer(FOV_diam / 2)
        return rect_polygon.intersection(aperture)

    return rect_polygon

# -------------------------------------------------------------------------------
