library(rethinking)
library(cmdstanr)
library(here)

d <- read.csv(here('results', 'variance_per_speaker.csv'))

dat  <- list(
	     S = standardize(d$r_s),
         F = standardize(d$r_fs),
	     T = standardize(d$tmp_var),
	     A = standardize(d$amp_var),
         N = d$subject
)


m <- ulam(
          alist(
                F ~ dnorm(mu, sigma),
                mu <- a[N]+bT*T+bA*A+bS*S, 
                a[N] ~ dnorm(a, 0.2), 
                bA ~ dnorm(0, 0.5), 
                bT ~ dnorm(0, 0.5), 
                bS ~ dnorm(0, 0.5), 
                sigma ~ dexp(1)
          ), data=dat)

mST <- ulam(
    alist(
        S ~ dbinom(D,p),
        logit(p) <- a[T],
        a[T] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1)
        ), data=dat, chains=4, log_lik=TRUE)

m <- quap(f, data=dat)
# posterior predictive plot --> how well does the model match
# the actually observed data
The column name or column position to be used as horizontal coordinates for each p
mu <- link(m) # call link using the original data
mu_mean <- apply(mu, 2, mean)
mu_PI <- apply(mu, 2, PI)
# simulate observations
W_sim <- sim(m, n=1e4)
W_pi <- apply(W_sim, 2, PI)

# plot the actually observed weights against the predicted weights
# useful for determining how good the model is
plot(mu_mean ~ dat$W, xlab='Observed weight', ylab='Predicted Weight')




