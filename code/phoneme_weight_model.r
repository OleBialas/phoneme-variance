library(rethinking)
library(cmdstanr)
library(here)

d <- read.csv(here('results', 'phoneme_weights.csv'))

dat  <- list(
	     W = standardize(d$weight),
	     Td = standardize(d$phase_var),
	     Sd = standardize(d$amplitude_var),
	     C = standardize(d$count),
         Sub = d$subject
)

f <- alist(
	   W ~ dstudent(1, mu, sigma),
	   mu <- a[Sub]+bC*C+bT*Td+bS*Sd,
	   a[Sub] ~ dnorm(0, 0.2),
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
d$weight_sim = mu_mean
# plot the actually observed weights against the predicted weights
plot(mu_mean ~ dat$W, xlab='Observed weight', ylab='Predicted Weight')

d$weight_sim <- mu_mean
write.csv(d, here('results', 'phoneme_weights_predictions.csv'))

post <- extract.samples(m)

d <- data.frame(
    count=post$bC,
    amplitude=post$bS,
    phase=post$bT)

write.csv(d, here('results', 'posterior_samples.csv'))









