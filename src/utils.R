
## Binary classification metrics

confusion.matrix <- function(y.true, y.prob, prob=TRUE) {
  if (prob) {
    y.pred <- ifelse(y.prob > 0.5, 1, 0)
  } else {
    y.pred <- y.prob
  }

  true.pos <- sum(y.pred == 1 & y.true == 1)
  false.pos <- sum(y.pred == 1 & y.true == 0)
  false.neg <- sum(y.pred == 0 & y.true == 1)
  true.neg <- sum(y.pred == 0 & y.true == 0)

  return(matrix(c(true.pos, false.pos, false.neg, true.neg), 2, 2))
}

accuracy.score <- function(y.true, y.prob, prob=TRUE) {
  correct <- sum(diag(confusion.matrix(y.true, y.prob, prob)))
  
  return(correct / length(y.true))
}

recall.score <- function(y.true, y.prob, prob=TRUE) {
  true.pos - confusion.matrix(y.true, y.prob, prob)[1, 1]
  false.neg - confusion.matrix(y.true, y.prob, prob)[1, 2]

  return(true.pos / (true.pos + false.neg))
}

specificity.score <- function(y.true, y.prob, prob=TRUE) {
  true.neg - confusion.matrix(y.true, y.prob, prob)[2, 2]
  false.pos - confusion.matrix(y.true, y.prob, prob)[2, 1]

  return(true.neg / (true.neg + false.pos))
}

balaccuracy.score <- function(y.true, y.prob, prob=TRUE) {
  return(0.5 * (recall.score(y.true, y.prob, prob=TRUE) +
                  specificity.score(y.true, y.prob, prob=TRUE)))
}


## Data generating process generation

binary1 <- function(n, p1=0.5, mu0=0, mu1=2, sigma=1) {
  set.seed(1)
  Y <- rbinom(n, 1, p1)
  # Y <- 2 * Y - 1
  # Y <- factor(Y)
  n1 <- sum(Y == 1)
  
  # if Y == 1 generate one sample from X|Y=1
  # otherwise generate one sample from X|Y=-1
  X <- matrix(NA, n, 1)
  X[Y == 1,] <- rnorm(n1, mu1, sigma)
  X[Y == 0,] <- rnorm(n - n1, mu0, sigma)
  
  return(as.data.frame(cbind(Y=Y, X1=X[,1])))
}
