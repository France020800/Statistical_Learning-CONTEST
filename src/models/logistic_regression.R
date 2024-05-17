## Library
library(glmnet)
library(ggplot2)

## Dataset
data("iris")
iris.binary <- iris[iris$Species != "setosa",]
iris.binary$Species = factor(iris.binary$Species)
p = ncol(iris.binary)

## Logistic Regression
set.seed(111)
fit.all = glm(Species ~., data = iris.binary, family = "binomial")
summary(fit.all)

## Save results
simple_predicted_prob <- predict(fit.all, type = "response")
simple_predicted_class <- ifelse(simple_predicted_prob > 0.5, "virginica", "versicolor")

## Plot the results
ggplot(iris.binary, aes(x = Sepal.Length, y = simple_predicted_prob, color = Species)) +
  geom_point() +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
  labs(title = "Predicted Probabilities from Logistic Regression",
       x = "Sepal Length",
       y = "Predicted Probability") +
  theme_minimal()

## Logistic regression with Elastic Regularization
response = as.numeric(iris.binary$Species) - 1 # Convert factors to 0 and 1
predictors <- as.matrix(iris.binary[, 1:4])

set.seed(111)
cv.model  = cv.glmnet(predictors, response, family = "binomial", alpha = 0.5)
plot(cv.model)

# Extract the best lambda value
best.lambda <- cv.model$lambda.min
best.lambda

reg.elastic_predicted_prob = predict(cv.model, newx = predictors, s = best.lambda, type = "response")
reg.elastic_predicted_class = ifelse(reg.elastic_predicted_prob > 0.5, "virginica", "versicolor")

{plot(cv.model,xvar="lambda",label=FALSE, col=c(rep("#876197", 5), rep("#006C71", p-5)))
  abline(h=0, col="#FF9E77")
  text(4, 1.5, "Relevant", col="#876197")
  text(4, -1, "Irrelevant", col="#006C71")
  abline(v=log(cv.model$lambda[c(10,50,90)]), lty="dashed", col="grey60")
}

# Plot the results
ggplot(iris.binary, aes(x = Sepal.Length, y = reg.elastic_predicted_prob, color = Species)) +
  geom_point() +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
  labs(title = "Predicted Probabilities from Regularized Logistic Regression",
       x = "Sepal Length",
       y = "Predicted Probability") +
  theme_minimal()

## Logistic regression with Lasso Regularization
response = as.numeric(iris.binary$Species) - 1 # Convert factors to 0 and 1
predictors <- as.matrix(iris.binary[, 1:4])

set.seed(111)
cv.model  = cv.glmnet(predictors, response, family = "binomial", alpha = 1)
plot(cv.model)

# Extract the best lambda value
best.lambda <- cv.model$lambda.min
best.lambda

reg.lasso_predicted_prob = predict(cv.model, newx = predictors, s = best.lambda, type = "response")
reg.lasso_predicted_class = ifelse(reg.elastic_predicted_prob > 0.5, "virginica", "versicolor")

## Logistic regression with Ridge Regularization
response = as.numeric(iris.binary$Species) - 1 # Convert factors to 0 and 1
predictors <- as.matrix(iris.binary[, 1:4])

set.seed(111)
cv.model  = cv.glmnet(predictors, response, family = "binomial", alpha = 0)
plot(cv.model)

# Extract the best lambda value
best.lambda <- cv.model$lambda.min
best.lambda

reg.ridge_predicted_prob = predict(cv.model, newx = predictors, s = best.lambda, type = "response")
reg.ridge_predicted_class = ifelse(reg.elastic_predicted_prob > 0.5, "virginica", "versicolor")

## Compare methods
simple.accuracy = mean(simple_predicted_class == iris.binary$Species)
reg.elastic.accuracy = mean(reg.elastic_predicted_class == iris.binary$Species)
reg.lasso.accuracy = mean(reg.lasso_predicted_class == iris.binary$Species)
reg.ridge.accuracy = mean(reg.ridge_predicted_class == iris.binary$Species)
cat("Simple Model Accuracy:", simple.accuracy, "\n")
cat("Elastic Model Accuracy:", reg.elastic.accuracy, "\n")
cat("Lasso Model Accuracy:", reg.lasso.accuracy, "\n")
cat("Ridge Model Accuracy:", reg.ridge.accuracy, "\n")
