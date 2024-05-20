
## Explore dataset

# class distribution
dataset.distrib <- function(y) {
  return(table(y) / length(y))
}


## Binary classification metrics

# confusion matrix
confusion.matrix <- function(y.true, y.prob, prob=TRUE) {
  if (prob) {
    y.pred <- ifelse(y.prob > 0.5, 1, -1)
  } else {
    y.pred <- y.prob
  }

  true.pos <- sum(y.pred == 1 & y.true == 1)
  false.pos <- sum(y.pred == 1 & y.true == -1)
  false.neg <- sum(y.pred == -1 & y.true == 1)
  true.neg <- sum(y.pred == -1 & y.true == -1)

  mat <- matrix(c(true.pos, false.pos, false.neg, true.neg), 2, 2)
  rownames(mat) <- colnames(mat) <- c("1", "-1")
  return(mat)
}

# accuracy
accuracy.score <- function(y.true, y.prob, prob=TRUE) {
  correct <- sum(diag(confusion.matrix(y.true, y.prob, prob)))
  
  return(correct / length(y.true))
}

# recall
recall.score <- function(y.true, y.prob, prob=TRUE) {
  true.pos <- confusion.matrix(y.true, y.prob, prob)[1, 1]
  false.neg <- confusion.matrix(y.true, y.prob, prob)[1, 2]

  return(true.pos / (true.pos + false.neg))
}

# specificity
specificity.score <- function(y.true, y.prob, prob=TRUE) {
  true.neg <- confusion.matrix(y.true, y.prob, prob)[2, 2]
  false.pos <- confusion.matrix(y.true, y.prob, prob)[2, 1]

  return(true.neg / (true.neg + false.pos))
}

# balanced accuracy
balaccuracy.score <- function(y.true, y.prob, prob=TRUE) {
  return(0.5 * (recall.score(y.true, y.prob, prob) +
                  specificity.score(y.true, y.prob, prob)))
}

# misclassification error rate
misclass.error <- function(y.true, y.prob, prob=TRUE) {
  N <- length(y.true)
  if (prob) {
    y.pred <- ifelse(y.prob > 0.5, 1, -1)
  } else {
    y.pred <- y.prob
  }

  return(sum(y.true != y.pred) / N)
}


## Ensemble metrics

# confusion matrix, accuracy, balanced accuracy
first.metrics <- function(train.true, train.prob, test.true, test.prob, prob=TRUE) {
  print(confusion.matrix(train.true, train.prob, prob))

  print(paste0("Train score: ", accuracy.score(train.true, train.prob, prob)))
  print(paste0("Train balanced score: ", balaccuracy.score(train.true, train.prob, prob)))

  print(paste0("Test score: ", accuracy.score(test.true, test.prob, prob)))
  print(paste0("Test balanced score: ", balaccuracy.score(test.true, test.prob, prob)))
}


## Cross-Validation

# adaboost cross-validation
adaboost.cv <- function(M.max, full.train, nfolds, response, lam=1, bag=1, seed=111) {
  target <- which(colnames(full.train) == response)

  myfolds <- cut(1:nrow(full.train), breaks=nfolds, labels=FALSE)

  folds.metrics <- matrix(NA, M.max, nfolds)

  # set.seed(seed)  # reproducibility due to bootstrapping
  for (i in 1:nfolds) {
    # reproducibility due to bootstrapping
    if (bag < 1) {set.seed(seed)}

    data.train <- full.train[myfolds != i,]  # training data
    data.val <- full.train[myfolds == i,]  # validation data

    # compute validation error for each fold for each round of boosting
    folds.metrics[,i] <- {
      # fit model, package ada
      mod.adaboost <- ada::ada(x=as.matrix(data.train[,-target]), y=as.factor(data.train[,target]),
                               test.x=as.matrix(data.val[,-target]), test.y=as.factor(data.val[,target]),
                               loss="exponential", type="discrete",
                               iter=M.max, nu=lam, bag.frac=bag)

      mod.adaboost$model$errs[,3]
    }

  }

  # compute error rate for each value in grid
  rowMeans(folds.metrics)
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
