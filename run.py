from src.train import Train

# Load diabetes data
train = Train()
X, y = train.load_data()

# Create an DataFrame
df_diabetes = train.create_dataframe(X, y, ['age',
                                            'sex',
                                            'bmi',
                                            'bp',
                                            's1',
                                            's2',
                                            's3',
                                            's4',
                                            's5',
                                            's6',
                                            'progression'
                                            ])

# Split data
train_data, test_data = train.split_data(df_diabetes)

# Train the model
model = train.train(train_data)

# Evaluate
train.evaluate(model, test_data)

# Predict!
print(train.predict(model, test_data.drop(["progression"], axis=1)))

# Persist the model
train.persist_model(model, 'model/lr-diabetes.model')
