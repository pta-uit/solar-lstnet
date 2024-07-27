import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Evaluator:
    def evaluate(self, y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        r2 = r2_score(y_true_flat, y_pred_flat)
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)

        return r2, mse, mae

    def plot_results(self, y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Plot predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
        plt.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
        plt.xlabel('Actual Solar Generation [W/kW]')
        plt.ylabel('Predicted Solar Generation [W/kW]')
        plt.title('Predicted vs Actual Solar Generation')
        plt.show()

        # Plot Actual vs Predicted Solar Generation
        plt.figure(figsize=(16, 8))
        step = 24 * 7 * 2  # Plot every 2 weeks
        time_steps = range(0, len(y_true_flat), step)
        plt.plot(time_steps, y_true_flat[::step], label='Actual', alpha=0.7)
        plt.plot(time_steps, y_pred_flat[::step], label='Predicted', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Solar Generation [W/kW]')
        plt.title('Actual vs Predicted Solar Generation (Test Dataset)')
        plt.legend()
        plt.show()

        # Plot a sample prediction
        sample_idx = np.random.randint(0, len(y_true))
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_true[sample_idx])), y_true[sample_idx], label='Actual')
        plt.plot(range(len(y_pred[sample_idx])), y_pred[sample_idx], label='Predicted')
        plt.title('Sample Prediction')
        plt.xlabel('Hours')
        plt.ylabel('Solar Generation [W/kW]')
        plt.legend()
        plt.show()