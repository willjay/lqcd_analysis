import numpy as np
import src.shrink as shrink

def test_pav():
        test_dat = np.array([
            0.78204, 0.78931, 0.71928, 0.88077, 0.97439,
            0.92744, 0.95181, 0.92588, 0.84977, 0.85741,
            0.84426, 0.95874, 1.10033, 1.08001, 1.06510,
            1.05460, 1.10427, 1.18246, 1.29133, 1.43503,
        ])
        v_expect = np.array([
            0.76354, 0.76354, 0.76354, 0.88077, 0.90442,
            0.90442, 0.90442, 0.90442, 0.90442, 0.90442,
            0.90442, 0.95874, 1.07501, 1.07501, 1.07501,
            1.07501, 1.10427, 1.18246, 1.29133, 1.43503,
        ])
        print("=== test pav() ===")
        v_try = shrink._pav(test_dat)
        if np.any(np.abs(v_try - v_expect) > 0.5e-5):
            print("FAIL")
        else:
            print("PASS")
            pass
        return


def test_nl_shrink_ngtp():
        n_test = 40
        test_sample_ev = np.array([
            0.13310, 0.16168, 0.19690, 0.27357, 0.33348,
            0.37520, 0.43858, 0.47237, 0.60298, 0.71604,
            0.89067, 1.03672, 1.08126, 1.19031, 1.30455,
            1.42597, 1.73527, 2.00243, 2.18337, 2.56383,
        ])
        dhat_expect = np.array([
            0.80632, 0.80632, 0.80632, 0.80632, 0.80632,
            0.80632, 0.80632, 0.80632, 0.89186, 0.95985,
            0.99774, 0.99774, 0.99774, 0.99774, 0.99774,
            0.99774, 0.99774, 1.09078, 1.21448, 1.40181,
        ])
        print("=== test direct_nl_shrink()  n>p ===")
        dhat_try = shrink.direct_nl_shrink(test_sample_ev, n_test)
        if np.any(np.abs(dhat_try - dhat_expect) > 2.0e-5):
            print('diff(dhat):\n', np.abs(dhat_try - dhat_expect))
            print("FAIL")
        else:
            print("PASS")
            pass
        return

def test_nl_shrink_nltp():
        n_test = 18
        test_sample_ev = np.array([
            1.5221e-16, 2.2853e-16, 2.5385e-02, 5.4003e-02, 8.1657e-02,
            1.2770e-01, 1.9568e-01, 2.9341e-01, 4.0082e-01, 5.5239e-01,
            6.7800e-01, 9.2593e-01, 1.0849e+00, 1.4008e+00, 1.5547e+00,
            1.7357e+00, 2.2198e+00, 2.5691e+00, 3.0961e+00, 3.4351e+00,
        ])
        dhat_expect = np.array([
            0.78890, 0.78890, 0.78890, 0.78890, 0.78890,
            0.78890, 0.78890, 0.91823, 1.00889, 1.00889,
            1.00889, 1.00889, 1.00889, 1.00889, 1.00889,
            1.01888, 1.03971, 1.11032, 1.46767, 1.51780,
        ])
        print("=== test direct_nl_shrink()  n<p ===")
        dhat_try = shrink.direct_nl_shrink(test_sample_ev, n_test)
        if np.any(np.abs(dhat_try - dhat_expect) > 5.0e-5):
            print('diff(dhat):\n', np.abs(dhat_try - dhat_expect))
            print("FAIL")
        else:
            print("PASS")
            pass
        return                