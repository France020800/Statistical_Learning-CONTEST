## Library
library(glmnet)
library(ggplot2)
source("utils.R")

## Dataset
diab.data = read.csv("../datasets/diabetes.csv")
diab.n <- nrow(diab.data)
preprocess.mean <- sapply(1:8, function(i) {mean(diab.data[,i])})
handle.idx <- c(2, 3, 4, 5, 6)

for (i in handle.idx) {
  diab.data[,i][diab.data[,i] == 0] <- preprocess.mean[i]
}

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
alpha = 0     # Lasso = 1 -- Ridge = 0 -- 0 < Elastic < 1
cv.model = cv.glmnet(x.train, y.train, family = 'binomial', alpha = alpha)
pdf("../plots/cv_lambda.pdf")
plot(cv.model)
dev.off()
best.lambda <- cv.model$lambda.min
best.lambda

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
pdf("../plots/diabetes_elasticnet.pdf")
{plot(cv.model$glmnet.fit, xvar = "lambda", label = FALSE, col = mypal, lwd = 3)
  abline(h=0, col="grey10")
  abline(v = log(cv.model$lambda[c(50, 100)]), lty = "dashed", col = "grey60")
  legend("topright",fill=mypal,rownames(cv.model$glmnet.fit$beta), xpd=FALSE, cex=0.8, bty = "n")
}
dev.off()

optimal_coefs <- coef(cv.model, s = "lambda.min")
coef_df <- as.data.frame(as.matrix(optimal_coefs))
coef_df$Variable <- rownames(coef_df)
names(coef_df)[1] <- "Coefficient"
coef_df <- coef_df[coef_df$Variable != "(Intercept)", ]
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

pdf("../plots/variable_importance_elasticnet.pdf")
{ggplot(data = coef_df, aes(x = reorder(Variable, abs(Coefficient)), y = abs(Coefficient))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # Flip coordinates to make it horizontal
  labs(title = "Variable Importance from ElasticNet",
       x = "Variables",
       y = "Absolute Coefficient Value") +
  theme_minimal()
}
dev.off()

{plot(cv.model$glmnet.fit, xvar = "norm", label = FALSE, col=mypal)
  legend("bottomleft",fill=mypal,rownames(coef(cv.model)), xpd=FALSE, cex=0.8, bty = "n")}

## RESULTS
# ElasticNet
# Train accuracy:  0.7636719 
# Test accuracy:  0.7773438
# Train accuracy:  0.7734375  // Modified dataset 
# Test accuracy:  0.7695312   // Modified dataset
# Lambda: 0.008591009
#
# Ridge
# Train accuracy:  0.7636719 
# Test accuracy:  0.78125
# Train accuracy:  0.7695312  // Modified dataset
# Test accuracy:  0.765625    // Modified dataset
# Lambda: 0.02346324
#
# Lasso
# Train accuracy:  0.765625 
# Test accuracy:  0.7734375 
# Train accuracy:  0.7714844  // Modified dataset
# Test accuracy:  0.7695312   // Modified dataset
# Lambda: 0.005678404
#
# Adalasso
# Train accuracy:  0.7753906 
# Test accuracy:  0.7734375 
# Lambda: 0.1417986

## ADALASSO
set.seed(111)
fit.0 = glm(Outcome ~., family = binomial, data = diab.train)
initial_coefs <- initial_coefs[-1]
weights <- 1 / abs(initial_coefs)
weights_matrix <- diag(weights)
cv.model <- cv.glmnet(x.train, y.train, family = "binomial", alpha = 1, penalty.factor = weights)
best.lambda <- cv.model$lambda.min

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

pdf("../plots/diabetes_adalasso.pdf")
{plot(cv.model$glmnet.fit, xvar = "lambda", label = FALSE, col = mypal, lwd = 3)
  abline(h=0, col="grey10")
  abline(v = log(cv.model$lambda[c(50, 100)]), lty = "dashed", col = "grey60")
  legend("topright",fill=mypal,rownames(cv.model$glmnet.fit$beta), xpd=FALSE, cex=0.8, bty = "n")
}
dev.off()


optimal_coefs <- coef(cv.model, s = "lambda.min")
coef_df <- as.data.frame(as.matrix(optimal_coefs))
coef_df$Variable <- rownames(coef_df)
names(coef_df)[1] <- "Coefficient"
coef_df <- coef_df[coef_df$Variable != "(Intercept)", ]
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

pdf("../plots/variable_importance_adalasso.pdf")
{ggplot(data = coef_df, aes(x = reorder(Variable, abs(Coefficient)), y = abs(Coefficient))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # Flip coordinates to make it horizontal
  labs(title = "Variable Importance from Adaptive Lasso",
       x = "Variables",
       y = "Absolute Coefficient Value") +
  theme_minimal()
}
dev.off()
