import jax
import time
from jax import jit
import jax.numpy as jnp

def predict(params: dict, X: jnp.array):
    return params["w"] @ X + params["b"]
    
def loss_fn(params: dict, X: jnp.array, y: jnp.array):
    y_hat = predict(params, X)
    return jnp.mean((y_hat - y)**2)

@jit
def train_step(params: dict, X: jnp.array, y: jnp.array, lr: float):
    loss = loss_fn(params, X, y) #only used for logging
    grads = jax.grad(loss_fn)(params, X, y)
    params_updated = {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"]
    }
    return params_updated, loss

def train(params, X, y, lr=0.01, epochs = 10):
    total_time_elapsed = 0
    for epoch in range(epochs):
        start = time.perf_counter()
        params, loss = train_step(params, X, y, lr)
        end = time.perf_counter()
        epoch_time = end - start
        total_time_elapsed += epoch_time
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Execution Time: {epoch_time:.4f}")
    return params, total_time_elapsed

def main():
    key = jax.random.PRNGKey(0)
    
    num_features = 3
    num_samples = 200
    X = jax.random.normal(key, (num_features, num_samples))
    
    # True weights and bias
    true_w = jnp.array([[2.0, -1.5, 0.5]])
    true_b = jnp.array([1.0])
    
    y = true_w @ X + true_b + 0.1 * jax.random.normal(key, (1, num_samples))
    
    params = {
        "w": jax.random.normal(key, (1, num_features)),
        "b": jnp.zeros((1,))
    }

    
    params, total_time_elapsed = train(params, X, y, lr=0.05, epochs=50)
    print("Training complete, total time taken:", total_time_elapsed)
    error_w = true_w - params["w"]
    error_b = true_b - params["b"]
    print(error_w)
    print(error_b)
    
    # Test
    X_test = jnp.array([[1.0, -1.0, 2.0]]).T  # shape (3,1)
    y_pred = predict(params, X_test)
    y_true =  true_w @ X_test + true_b
    print(f"Prediction for {X_test.T}: {y_pred[0,0]:.4f}")
    print("True value", y_true)

if __name__ == "__main__":
    main()
