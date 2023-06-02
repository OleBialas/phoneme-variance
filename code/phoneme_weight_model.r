library(rethinking)
library(cmdstanr)
library(here)

d <- read.csv(here('results', 'phoneme_weights.csv'))

dat  <- list(
	     W = standardize(d$weight),
	     Td = standardize(d$tvar),
	     Sd = standardize(d$svar),
	     C = standardize(d$count),
         Sub = d$subject
)

f <- alist(
	   W ~ dstudent(1, mu, sigma),
	   mu <- a+bC*C+bT*Td+bS*Sd,
	   a ~ dnorm(0, 0.2),
	   bC ~ dnorm(0, 0.5),
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
plot(mu_mean ~ dat$W, xlab='Observed weight', ylab='Predicted Weight', xlim=c(-1.5,4.5), ylim=c(-1.0, 1.0))
abline(a=0, b=1)




