import xgboost
import shap

# train an XGBoost model
X, y = shap.datasets.california()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0])

# visualize all the training set predictions
shap.plots.force(shap_values[:500])

# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:, "Latitude"], color=shap_values)



# summarize the effects of all the features
shap.plots.beeswarm(shap_values)




#http://github.com/shap/shap