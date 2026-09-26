"""Run with python check_app.py; no private data required."""
import numpy as np
from analysis import synthetic_demo, describe, cluster_profiles
from streamlit.testing.v1 import AppTest

e,_,_=synthetic_demo()
df,x=describe(e)
l,centers,sil,ari=cluster_profiles(x,4)
assert len(df)==80 and np.isfinite(x).all()
assert sorted(np.bincount(l))==[20]*4
at=AppTest.from_file('app.py',default_timeout=60).run()
assert not at.exception
at.sidebar.radio[0].set_value('Synthetic demonstration').run()
assert not at.exception
at.button[0].click().run()
assert not at.exception
at.slider[0].set_value(3).run()
assert not at.exception
print('Demo grouping, app startup, group exploration and setting changes passed.')
