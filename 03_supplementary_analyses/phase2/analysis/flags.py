# -*- coding: utf-8 -*-
"""Traffic-light flags derived from the midterm score (user ruling R5): red <= 50, yellow 51-59, green >= 60.
Values strictly between 50 and 51 do not occur in the data; they are treated as yellow (> 50 and < 60)."""


def traffic_light(midterm):
    m = float(midterm)
    if m <= 50:
        return 'red'
    if m < 60:
        return 'yellow'
    return 'green'
