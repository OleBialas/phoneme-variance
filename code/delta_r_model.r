library(rethinking)
library(cmdstanr)
library(here)

d <- read.csv(here('results', 'variance_per_speaker.csv'))

dat  <- list(
	     R = standardize(d$delta_r),
	     Td = standardize(d$temp_var),
	     Sd = standardize(d$amp_var),
         Sub = d$subject_nr
)

f <- alist(
	   R ~ dnorm(mu, sigma),
	   mu <- a[Sub]+bT*Td+bS*Sd,
	   a[Sub] ~ dnorm(0, 0.2),
	   bT ~ dnorm(0, 0.5),
	   bS ~ dnorm(0, 0.5),
	   sigma ~ dexp(1)
)

m <- quap(f, data=dat)
# posterior predictive plot --> how well does the model match
# the actually observed data

mu <- link(m) # call link using the original data
mu_mean <- apply(mu, 2, mean)
mu_PI <- apply(mu, 2, PI)
# simulate observations
W_sim <- sim(m, n=1e4)
W_pi <- apply(W_sim, 2, PI)

# plot the actually observed weights against the predicted weights
# useful for determining how good the model is
plot(mu_mean ~ dat$W, xlab='Observed weight', ylab='Predicted Weight')




