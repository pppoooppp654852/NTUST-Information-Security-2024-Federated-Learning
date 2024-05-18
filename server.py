import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import EvaluateRes, FitRes, Scalar, Metrics
import json

record_results = {"loss": [], "accuracy": []}

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, Exception]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""
        
        if not results:
            return None, {}

        # Initialize sums for accuracy and loss
        total_accuracy = 0.0
        total_loss = 0.0
        total_examples = 0
        
        # Accumulate weighted accuracy and loss from each client
        for client_proxy, evaluate_res in results:
            metrics = evaluate_res.metrics
            num_examples = metrics.get("num_examples", 0)
            total_examples += num_examples

            if "accuracy" in metrics and "loss" in metrics:
                accuracy = metrics["accuracy"]
                loss = metrics["loss"]
                total_accuracy += accuracy * num_examples
                total_loss += loss * num_examples

        # Compute weighted averages
        if total_examples > 0:
            aggregated_accuracy = total_accuracy / total_examples
            aggregated_loss = total_loss / total_examples
        else:
            aggregated_accuracy = 0.0
            aggregated_loss = 0.0

        print(f"Round {server_round}: Aggregated Accuracy: {aggregated_accuracy}, Aggregated Loss: {aggregated_loss}")
        
        record_results["loss"].append(aggregated_loss)
        record_results["accuracy"].append(aggregated_accuracy)
        
        # Return loss (None if not calculated) and aggregated metrics
        return aggregated_loss, {"accuracy": aggregated_accuracy, "loss": aggregated_loss}

# Initialize and start the server with the custom strategy
strategy = AggregateCustomMetricStrategy()
fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=100), strategy=strategy)

save_path = "results/federated_learning.json"
with open(save_path, "w") as file:
    json.dump(record_results, file)
print("Results have been saved as", save_path)