from hmmlearn.base import _BaseHMM
import numpy as np
import sklearn.cluster as cluster
from sklearn.utils import check_random_state
from scipy.special import gammaln
import matplotlib.pyplot as plt


def log_multivariate_poisson_density(X, means):
    # ------------------------------------------------------------------------------------------------------------------
    # modeled on log_multivariate_normal_density from sklearn.mixture
    #
    # params:   - X, array: data
    #           - means, array: means of all modes [modes, cells]
    #
    # ------------------------------------------------------------------------------------------------------------------
    n_samples, n_dim = X.shape
    # -lambda + k log(lambda) - log(k!)
    means = np.nan_to_num(means)
    # TODO: why do i have negative lambdas???
    means[means<0] = 1e-3
    means[means == 0] = 1e-3

    log_means = np.where(means > 0, np.log(means), np.log(1e-3))
    lpr = np.dot(X, log_means.T)
    lpr = lpr - np.sum(means,axis=1) # rates for all elements are summed and then broadcast across the observation dimenension
    log_factorial = np.sum(gammaln(X + 1), axis=1)
    lpr = lpr - log_factorial[:,None] # logfactobs vector broad cast across the state dimension

    return lpr

class PoissonHMM(_BaseHMM):
    """Hidden Markov Model with independent Poisson emissions.
    Parameters
    ----------
    n_components : int
        Number of states.
    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.
    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.
    algorithm : string, one of the :data:`base.DECODER_ALGORITHMS`
        Decoder algorithm.
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.
    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.
    Attributes
    ----------
    n_features : int
        Dimensionality of the (independent) Poisson emissions.
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.
    Examples
    --------
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PoissonHMM(algorithm='viterbi',...
    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.means_prior = means_prior
        self.means_weight = means_weight

    def _check(self):
        super(PoissonHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

    def _compute_log_likelihood(self, obs):
        return log_multivariate_poisson_density(obs, self.means_)

    def _generate_sample_from_state(self, state, random_state=None):
        rng = check_random_state(random_state)
        return rng.poisson(self.means_[state])

    def _init(self, X, lengths=None, params='stmc'):
        super(PoissonHMM, self)._init(X, lengths=lengths)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats):
        super(PoissonHMM, self)._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))
            self.means_ = np.where(self.means_ > 1e-5, self.means_, 1e-3)


def simulate_poisson_spiking(nr_neurons, time_bins, nr_states):

    lambda_states = np.random.randint(0,10, size=((nr_neurons, nr_states)))

    mean_firing = []
    state_seq = np.zeros(time_bins)

    for i in range(time_bins):
        state = np.random.randint(0, nr_states)
        mean_firing.append(lambda_states[:, state])
        state_seq[i] = state

    mean_firing = np.array(mean_firing).T
    poisson_firing = np.random.poisson(mean_firing)

    return poisson_firing, state_seq


if __name__ == '__main__':

    X, state_seq = simulate_poisson_spiking(nr_neurons=100, time_bins=1000, nr_states=5)
    X_train= X[:, :int(X.shape[1]*0.5)]
    X_test = X[:, int(X.shape[1] * 0.5):]

    plt.subplot(2,1,1)
    plt.imshow(X, interpolation='nearest', aspect='auto')
    plt.subplot(2, 1, 2)
    plt.imshow(np.expand_dims(state_seq, axis=0), interpolation='nearest', aspect='auto')
    plt.show()

    logli = []
    for i in range(1, 20):
        model = PoissonHMM(n_components=i)
        model.fit(X_train.T)
        logli.append(model.score(X_test.T))

    nr_modes = np.arange(1, len(logli)+1)
    plt.plot(nr_modes[4:], logli[4:])
    plt.grid()
    plt.show()

    plt.imshow(model.transmat_, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.show()

    samples, state_sequence = model.sample(1000)
    print(samples)
    plt.imshow(samples, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.show()


    Z2 = model.predict(X_test.T)
    plt.subplot(2, 1, 1)
    plt.imshow(np.expand_dims(state_seq[int(X.shape[1] * 0.5):], axis=0), interpolation='nearest', aspect='auto')
    plt.subplot(2, 1, 2)
    plt.imshow(np.expand_dims(Z2, axis=0), interpolation='nearest', aspect='auto')
    plt.show()
    a = model.score(X_test.T)
    print(a)
