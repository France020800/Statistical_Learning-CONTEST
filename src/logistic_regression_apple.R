## Library
library(glmnet)
library(ggplot2)
source("utils.R")

## Dataset
apple.data = read.csv("../datasets/apple_quality.csv")
apple.data = na.omit(apple.data)
apple.data$Acidity = as.numeric(apple.data$Acidity)
apple.n <- nrow(apple.data)
apple.test.n <- floor(apple.n / 3)

set.seed(111)
apple.train.idx <- sample(apple.n, apple.n - apple.test.n, replace=FALSE)

apple.train <- apple.data[apple.train.idx,]
apple.test <- apple.data[-apple.train.idx,]

x.train = as.matrix(apple.train[, 1:8])
y.train = ifelse(apple.train$Quality == 'bad', 0, 1)
x.test = as.matrix(apple.test[, 1:8])
y.test = ifelse(apple.test$Quality == 'bad', 0, 1)

## Regularization 
alpha = 0.5     # Lasso = 1 -- Ridge = 0 -- 0 < Elastic < 1
cv.model = cv.glmnet(x.train, y.train, family = 'binomial', alpha = alpha)
best.lambda <- cv.model$lambda.min
best.lambda
plot(cv.model)

# Metrics
predicted_probabilities <- predict(cv.model, newx = x.train, s = "lambda.min", type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
accuracy = accuracy.score(y.train, predicted_classes)
recall = recall.score(y.train, predicted_classes)
specificity = specificity.score(y.train, predicted_classes)
cat("Train accuracy: ", accuracy, "\n")
cat("Train recall: ", recall, "\n")
cat("Train specificity: ", specificity, "\n")

predicted_probabilities <- predict(cv.model, newx = x.test, s = "lambda.min", type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
accuracy = accuracy.score(y.test, predicted_classes)
recall = recall.score(y.test, predicted_classes)
specificity = specificity.score(y.test, predicted_classes)
cat("Test accuracy: ", accuracy, "\n")
cat("Test recall: ", recall, "\n")
cat("Test specificity: ", specificity, "\n")

# Plot the regularization path
mypal = c(RColorBrewer::brewer.pal(12,"Set3"), RColorBrewer::brewer.pal(6,"Dark2"))
{plot(cv.model$glmnet.fit, xvar = "lambda", label = FALSE, col = mypal)
abline(h=0, col="grey10")
abline(v = log(cv.model$lambda[c(50, 100)]), lty = "dashed", col = "grey60")
legend("topright",fill=mypal,rownames(cv.model$glmnet.fit$beta), xpd=FALSE, cex=0.8, bty = "n")
}

## RESULTS
# Lasso - Train acc: 0.7416573 Test acc: 0.7576894 
# Ridge - Train acc: 0.7431571 Test acc: 0.7591898 
# ElasticNet - Train acc: 0.7416573 Test acc: 0.7576894 
# Adalasso - Train acc: Test acc: 


## ADALASSO
fit.0 = glm(y.train ~ x.train, family = binomial)
initial_coefs <- coef(fit.0)
initial_coefs <- initial_coefs[-1]
weights <- 1 / abs(initial_coefs)
weights_matrix <- diag(weights)
cv.model <- cv.glmnet(x.train, y.train, family = "binomial", alpha = 1, penalty.factor = weights)
print(coef(cv.model, s = "lambda.min"))

# Accuracy
predicted_probabilities <- predict(cv.model, newx = x.train, s = "lambda.min", type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
print(table(Predicted = predicted_classes, Actual = y.train))

accuracy = accuracy.score(y.train, predicted_classes)
cat("Train accuracy: ", accuracy, "\n")

predicted_probabilities <- predict(cv.model, newx = x.test, s = "lambda.min", type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
print(table(Predicted = predicted_classes, Actual = y.test))
accuracy = accuracy.score(y.test, predicted_classes)
cat("Train accuracy: ", accuracy, "\n")


