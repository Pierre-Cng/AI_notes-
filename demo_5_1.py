from keras import models, layers 

from demo_5 import build_model
from demo_5 import train_data, train_targets, test_data, test_targets

model = build_model()
model.fit(train_data, train_targets,
epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)