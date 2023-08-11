library(rethinking)
library(cmdstanr)
library(here)

d <- read.csv(here('results', 'phoneme_weights.csv'))

dat  <- list(
	     W = standardize(d$weight),
	     P = standardize(d$phase_var),
	     A = standardize(d$amplitude_var),
	     O = standardize(d$occurrence),
         S = d$subject,
         C = standardize(d$correlation)
)

f <- alist(
	   W ~ dstudent(1, mu, sigma),
	   mu <- a[S]+bP*P+bA*A+bO*O+bC*C,
	   a[S] ~ dnorm(0, 0.2),
	   bP ~ dnorm(0, 0.5),
	   bA ~ dnorm(0, 0.5),
	   bO ~ dnorm(0, 0.5),
	   bC ~ dnorm(0, 0.5),
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
plot(mu_mean ~ dat$W, xlab='Observed weight', ylab='Predicted Weight')

d$weight_sim <- mu_mean
write.csv(d, here('results', 'phoneme_weights_predictions.csv'))

post <- extract.samples(m)

d <- data.frame(
    occurrence=post$bO,
    amplitude=post$bA,
    phase=post$bP,
    correlation=post$bC)

write.csv(d, here('results', 'posterior_samples.csv'))









