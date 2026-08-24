"""Vendored fastplotlib pieces the ndwidget branch dropped.

The masknmf stack requires fastplotlib's ndwidget branch, which removed
the classic ImageWidget and EdgeWindow the Studio viewer is built on.
This package carries the branch's own (last-working) ImageWidget +
slider UI, patched onto the branch's current APIs. Used only through
``mbo_utilities.gui._fpl_compat`` when the installed fastplotlib lacks
an importable ImageWidget.
"""
