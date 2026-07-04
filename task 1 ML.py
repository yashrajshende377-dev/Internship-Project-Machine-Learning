# Quick start example
predictor = HousePricePredictor()
data = predictor.load_sample_data()
X, y = predictor.prepare_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
predictor.train(X_train, y_train)
metrics, _ = predictor.evaluate(X_test, y_test)

# Predict a new house
new_house = pd.DataFrame({
    'square_footage': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})
price = predictor.predict(new_house)
print(f"Predicted price: ${price[0]:,.2f}")