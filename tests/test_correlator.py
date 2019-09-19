import os
import h5py
import sys
import gvar as gv
import pytest
# sys.path.append('/Users/wijay/GitHub/lqcd_analysis/')
import src.correlator as correlator


def main():

    test_BaseTimes()
    test_NPoint()


def test_main():
    """Test the main function."""
    correlator.main()


def test_BaseTimes():
    """Test correlator.BaseTimes class."""
    tdata = range(100)
    tmin = 1
    tmax = 50
    nt = 200
    tp = -1

    times = correlator.BaseTimes(tdata)
    print(times)
    assert times.tmin == 5, "Expected default tmin=5"
    assert times.tmax == len(tdata) - 1, "Expected default tmax=len(tdata)-1"
    assert times.nt == len(tdata), "Expected default nt=len(tdata)"
    assert times.tp == times.nt, "Expected default tp=nt"

    times = correlator.BaseTimes(tdata, tmin)
    assert times.tmin == tmin, "Expected tmin=1 set by hand"
    assert times.tmax == len(tdata) - 1, "Expected default tmax=len(tdata)-1"
    assert times.nt == len(tdata), "Expected default nt=len(tdata)"
    assert times.tp == times.nt, "Expected default tp=nt"

    times = correlator.BaseTimes(tdata, tmin, tmax)
    assert times.tmin == tmin, "Expected tmin=1 set by hand"
    assert times.tmax == tmax, "Expected tmin=50 set by hand"
    assert times.nt == len(tdata), "Expected default nt=len(tdata)"
    assert times.tp == times.nt, "Expected default tp=nt"

    times = correlator.BaseTimes(tdata, tmin, tmax, nt)
    assert times.tmin == tmin, "Expected tmin=1 set by hand"
    assert times.tmax == tmax, "Expected tmin=50 set by hand"
    assert times.nt == nt, "Expected nt=200 set by hand"
    assert times.tp == times.nt, "Expected default tp=nt"

    times = correlator.BaseTimes(tdata, tmin, tmax, nt, tp)
    assert times.tmin == tmin, "Expected tmin=1 set by hand"
    assert times.tmax == tmax, "Expected tmin=50 set by hand"
    assert times.nt == nt, "Expected nt=200 set by hand"
    assert times.tp == tp, "Expected tp=-1 set by hand"

    with pytest.raises(ValueError):
        times = correlator.BaseTimes(tdata, tmin=-1)

    with pytest.raises(ValueError):
        times = correlator.BaseTimes(tdata, tmax=len(tdata)+1)


def test_NPoint():
    """Test correlator.TwoPoint and correlator.ThreePoint class."""
    basedir = '/Users/wijay/GitHub/lqcd_analysis/data/'
    h5fname = os.path.join(basedir, 'example_data.hdf5')
    data = read_example_data(h5fname)

    # Test for 'light-light' -- data is good everywhere
    tag = 'light-light'
    _, nt = data[tag].shape
    corr = gv.dataset.avg_data(data.pop(tag))
    c2_src = correlator.TwoPoint(tag, corr)
    print(c2_src)
    assert len(c2_src) == nt,\
        "Unexpected len(c2)"
    assert len(c2_src[:]) == nt,\
        "Unexpected len(c2[:])"
    assert c2_src.times.tmax == (len(c2_src) - 1),\
        "Unexpected c2.times.tmax"
    assert len(c2_src.meff(avg=False)) == (len(c2_src) - 2),\
        "Unexpected len(c2_src.meff())"
    assert len(c2_src.meff(avg=True)) == (len(c2_src.times.tfit) - 4),\
        "Unexpected len(c2_src.meff())"
    assert len(c2_src.times.tfit) == (c2_src.times.tmax-c2_src.times.tmin),\
        "Unexpected len(c2_src.times.tfit)"
    assert len(c2_src.avg()) == (len(c2_src.times.tfit) - 2),\
        "Unexpected len(c2_src.avg())"
    assert c2_src.mass > 0,\
        "Invalid mass"
    assert c2_src.mass_avg > 0,\
        "Invalid mass from averaged correlator"

    # c2.__setitem__
    c2_src[0] = 1.0

    # Figures
    _ = c2_src.plot_corr(avg=False)
    _ = c2_src.plot_corr(avg=True)
    _ = c2_src.plot_meff(avg=False)
    _ = c2_src.plot_meff(avg=True)

    # Test for 'heavy-light' -- data is only good out to t=38
    tag = 'heavy-light'
    _, nt = data[tag].shape
    corr = gv.dataset.avg_data(data.pop(tag))
    c2_snk = correlator.TwoPoint(tag, corr)
    print(c2_snk)
    assert len(c2_snk) == nt,\
        "Unexpected len(c2_snk)"
    assert len(c2_snk[:]) == nt,\
        "Unexpected len(c2_snk[:])"
    assert c2_snk.times.tmax == 38,\
        "Unexpected c2_snk.times.tmax"
    assert len(c2_snk.meff()) == (len(c2_snk) - 2),\
        "Unexpected len(c2_snk.meff())"
    assert len(c2_snk.times.tfit) == (c2_snk.times.tmax-c2_snk.times.tmin),\
        "Unexpected len(c2_snk.times.tfit)"
    assert len(c2_snk.avg()) == (len(c2_snk.times.tfit) - 2),\
        "Unexpected len(c2_snk.avg())"
    assert c2_snk.mass > 0,\
        "Invalid mass"
    assert c2_snk.mass_avg > 0,\
        "Invalid mass from averaged correlator"

    _ = c2_snk.plot_corr(avg=False)
    _ = c2_snk.plot_corr(avg=True)
    _ = c2_snk.plot_meff(avg=False)
    _ = c2_snk.plot_meff(avg=True)
    
    # Test for three-point correlator
    tag = 'three-point correlator'
    ds = {key: gv.dataset.avg_data(val) for key, val in data.items()}
    c3 = correlator.ThreePoint(tag, ds)
    assert c3.times.tmax == 46,\
        "Unexpected c3.times.tmax"
    assert len(c3.times.tfit) == (46 - 5),\
        "Unexpected c3.times.tfit"
    assert len(c3) == len(ds),\
        "Unexpected len(c3)"
    
    # c3.__str__
    print(c3)
    
    avg = c3.avg(m_src=c2_src.mass, m_snk=c2_snk.mass)
    for val in avg.values():
        assert len(val) == (len(c3.times.tfit) - 2)

    with pytest.raises(TypeError):
        bad_data = {'broken': range(10)}
        c3 = correlator.ThreePoint("broken", bad_data)

    with pytest.raises(ValueError):
        bad_data = {1: range(10), 2: range(20)}
        c3 = correlator.ThreePoint("broken", bad_data)

    # c3.__setitem__ and c3.__get__item
    c3[13] = c3[14]

    # c3.__iter__
    for key in c3:
        pass

    # c3.items()
    for key, val in c3.items():
        pass


def read_example_data(h5fname):
    """Read the example data"""
    data = {}
    with h5py.File(h5fname, 'r') as ifile:
        dset = ifile['data']
        for key in dset.keys():
            data[key] = ifile['data'][key][:]
    for key in ['13','14','15','16']:
        data[int(key)] = data.pop(key)

    return data


if __name__ == '__main__':
    main()
