# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Parameters(models.Model):
	### The following 11 parameters can be overwritten by users
	# limits
	maxlines = models.IntegerField(default=300, help_text="maximum # lines permitted")
	# scale parameters
	scale = models.FloatField(default=0.0, help_text="the basic scale of the document (roughly, xheight) 0=automatic")
	hscale = models.FloatField(default=1.0, help_text="non-standard scaling of horizontal parameters")
	vscale = models.FloatField(default=1.0, help_text="non-standard scaling of vertical parameters")
	# line parameters
	threshold = models.FloatField(default=0.2, help_text="baseline threshold")
	noise = models.IntegerField(default=8, help_text="noise threshold for removing small components from lines")
	usegause = models.BooleanField(default=True, help_text="use gaussian instead of uniform")
	# column separator parameters
	maxseps = models.IntegerField(default=2, help_text="maximum # black column separators")
	sepwiden = models.IntegerField(default=10, help_text="widen black separators (to account for warping)")
	maxcolseps = models.IntegerField(default=3, help_text="maximum # whitespace column separators")
	csminheight = models.FloatField(default=10.0, help_text="minimum column height (units=scale)")

	### The following parameters needn't be overwritten by users
	# limits
	minscale = models.FloatField(default=1.0, help_text="minimum scale permitted")
	# output parameters
	pad = models.IntegerField(default=3, help_text="adding for extracted lines")
	expand = models.IntegerField(default=3, help_text="expand mask for grayscale extraction")
	# other parameters
	quiet = models.BooleanField(default=False, help_text="be less verbose, usally use with parallel together")
	nocheck = models.BooleanField(default=True, help_text="disable error checking on inputs")
	parallel = models.IntegerField(default=0, help_text="number of parallel processes to use")