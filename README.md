# lqcd_analysis
Analysis of lattice QCD data

TODO:
 * Add support for logger
 * Add code for hdf5 caching of correlators
 * Refactor and add code for database I/O
 * Add code for "serialization" of fits into the database
 * Add and refactor code for processing an embedded database into
   a centralized Postgres database
 * Verify that the refactored correlator code (meff, avg, etc...) works properly
   with folded correlators

DONE:
 * Profile, optimize, and clean up shrinkage code (this has become a
    bottleneck in production runs). The problem was that apparently np.einsum 
    was unexpectedly very slow.