## Library
library(glmnet)
library(ggplot2)

## Dataset
diab.data = read.csv("../../Materiale/diabetes.csv")
diab.n <- nrow(diab.data)
diab.test.n <- floor(diab.n / 3)

set.seed(111)
diab.train.idx <- sample(diab.n, diab.n - diab.test.n, replace=FALSE)

diab.train <- diab.data[diab.train.idx,]
diab.test <- diab.data[-diab.train.idx,]

x.test = as.matrix(diab.test[, 1:8])
y.test = diab.test$Outcome
x.train = as.matrix(diab.train[, 1:8])
y.train = diab.train$Outcome

## Regularization
set.seed(111)
alpha = 1     # Lasso = 1 -- Ridge = 0 -- 0 < Elastic < 1
cv.model = cv.glmnet(x.train, y.train, family = 'binomial', alpha = alpha)
best.lambda <- cv.model$lambda.min
best.lambda

# Accuracy
predicted_probabilities <- predict(cv.model, newx = x.train, s = "lambda.min", type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
accuracy = accuracy.score(y.train, predicted_classes)
cat("Train accuracy: ", accuracy, "\n")

predicted_probabilities <- predict(cv.model, newx = x.test, s = "lambda.min", type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
accuracy = accuracy.score(y.test, predicted_classes)
cat("Test accuracy: ", accuracy, "\n")

# Plot the regularization path
mypal = c(RColorBrewer::brewer.pal(12,"Set3"), RColorBrewer::brewer.pal(6,"Dark2"))
{plot(cv.model$glmnet.fit, xvar = "lambda", label = FALSE, col = mypal)
  abline(h=0, col="grey10")
  abline(v = log(cv.model$lambda[c(50, 100)]), lty = "dashed", col = "grey60")
  legend("topright",fill=mypal,rownames(cv.model$glmnet.fit$beta), xpd=FALSE, cex=0.8, bty = "n")
}

## RESULTS
# ElasticNet
# Train accuracy:  0.7636719 
# Test accuracy:  0.7773438
#
# Ridge
# Train accuracy:  0.7636719 
# Test accuracy:  0.78125
#
# Lasso
#
#