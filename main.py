import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def plot_metrics(precision, recall, f1):
    # Create a bar chart to display precision, recall, and F1 score
    labels = ['Precision', 'Recall', 'F1 Score']
    scores = [precision, recall, f1]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores, color=['blue', 'orange', 'green'])
    
    # Add labels and title
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score')
    
    # Display the score values on top of each bar
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.2f}', ha='center', fontsize=12)
    
    # Show the plot
    plt.show()
def calculate_metrics(recommendations, ground_truth):
    all_precisions = []
    all_recalls = []

    for customer_id in recommendations:
        y_true = list(ground_truth[customer_id])
        y_pred = recommendations[customer_id]
        
        # Convert to binary (1 if recommended, 0 otherwise)
        y_true_bin = [1 if product in y_true else 0 for product in y_pred]
        y_pred_bin = [1] * len(y_pred)  # Always recommending products

        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
    
    avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, f1

# Split the dataset into training and testing datasets
def split_data(data, test_size=0.2):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

def generate_recommendations_and_ground_truth(graph, test_data, top_n=5):
    recommendations = {}
    ground_truth = {}
    for customer_id in test_data['CustomerID'].unique():
        # Check if the customer exists in the graph
        if customer_id not in graph:
            print(f"Customer {customer_id} not found in graph, skipping recommendation.")
            continue  # Skip the customer if not found in the graph
        
        true_products = test_data[test_data['CustomerID'] == customer_id]['ProductID'].unique()
        recommended_products = recommend_products(graph, customer_id, top_n)
        
        recommendations[customer_id] = recommended_products
        ground_truth[customer_id] = set(true_products)
    
    return recommendations, ground_truth


# 1. Load Dataset with Row Limiting
def load_dataset(filepath, rows=1005):
    data = pd.read_csv(filepath)  # Replace with your dataset path
    data = data.head(rows)  # Limit to the first `rows` rows
    return data

# 2. Preprocess Data
def preprocess_data(data):
    # Select necessary columns and drop missing values
    data = data[['Customer ID', 'Product Category', 'Quantity']].dropna()

    # Rename columns for easier handling
    data.rename(columns={
        'Customer ID': 'CustomerID',
        'Product Category': 'ProductID',
        'Quantity': 'PurchaseFrequency'
    }, inplace=True)
    
    return data

# 3. Build Graph
def build_graph(data):
    G = nx.Graph()
    for _, row in data.iterrows():
        customer_id = row['CustomerID']
        product_id = row['ProductID']
        purchase_freq = row['PurchaseFrequency']

        # Ensure customers and products are in the graph
        if not G.has_node(customer_id):
            G.add_node(customer_id, type='customer')
        if not G.has_node(product_id):
            G.add_node(product_id, type='product')

        # Add an edge with weight representing purchase frequency
        G.add_edge(customer_id, product_id, weight=purchase_freq)

    return G

# 4. Depth-Limited BFS to Recommend Products
def recommend_products(graph, customer_id, top_n=5, max_depth=2):
    """
    Recommends products to a given customer based on graph traversal with depth-limitation.
    
    Parameters:
        graph (nx.Graph): The customer-product graph.
        customer_id (str): The ID of the customer.
        top_n (int): The number of recommendations to return.
        max_depth (int): Maximum depth of BFS traversal.
    
    Returns:
        list: Recommended products.
    """
    if customer_id not in graph:
        print(f"Customer {customer_id} not in graph.")
        return []  # Optionally provide a fallback recommendation, e.g., popular products

    # BFS Initialization
    visited = set()
    queue = deque([(customer_id, 0)])  # (node, current_depth)
    neighbors = []

    while queue:
        node, depth = queue.popleft()
        if depth < max_depth:
            # Explore the neighbors
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

                    # Check if the neighbor is a product (based on node type)
                    if graph.nodes[neighbor].get('type') == 'product':
                        neighbors.append(neighbor)
    
    if not neighbors:
        print(f"No recommendations found for customer {customer_id}.")
        return []  # Optionally return popular products as fallback

    # Count occurrences of each product (simulate popularity)
    product_counts = {}
    for product in neighbors:
        product_counts[product] = product_counts.get(product, 0) + 1

    # Sort products by frequency and return the top N
    sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
    return [product[0] for product in sorted_products[:top_n]]


# 5. Generate Recommendations for All Customers
def recommend_for_all_customers(graph, data, top_n=5):
    recommendations = {}
    for i, customer_id in enumerate(data['CustomerID'].unique()):
        if i % 10 == 0:  # Print progress every 10 customers
            print(f"Processing Customer {customer_id}...")
        recommended_products = recommend_products(graph, customer_id, top_n)
        recommendations[customer_id] = recommended_products
    return recommendations

# 6. Print Recommendations for All Customers in a Clear Format
def display_recommendations(recommendations):
    for customer_id, recs in recommendations.items():
        if recs:
            print(f"Customer {customer_id} - Recommended Products: {', '.join(map(str, recs))}")
        else:
            print(f"Customer {customer_id} - Recommended Products: No recommendations available")

# 7. Visualize the Entire Graph
def visualize_graph(graph):
    plt.figure(figsize=(12, 8))
    
    # Define node colors based on type
    node_colors = []
    for node in graph.nodes():
        if graph.nodes[node].get('type') == 'customer':
            node_colors.append('lightgreen')  # Color for customers
        elif graph.nodes[node].get('type') == 'product':
            node_colors.append('lightblue')  # Color for products

    # Visualize the entire graph (no node limit)
    pos = nx.spring_layout(graph, k=0.15, iterations=20)  # Positioning of nodes for better clarity
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, alpha=0.7)
    plt.title("Visualization of Customer-Product Graph (Entire Graph)")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "./ecommerce_customer_data_custom_ratios.csv"  # Update with your file location
    data = load_dataset(dataset_path, rows=1005)  # Limit to 1005 rows
    processed_data = preprocess_data(data)
    
    # Split data into training and testing sets
    train_data, test_data = split_data(processed_data)

    # Build graph using training data
    G_train = build_graph(train_data)

    # Build graph from processed data
    G = build_graph(processed_data)

    # Generate recommendations for the test dataset
    recommendations, ground_truth = generate_recommendations_and_ground_truth(G_train, test_data)
    
    # Calculate and display evaluation metrics
    precision, recall, f1 = calculate_metrics(recommendations, ground_truth)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

   

    # Visualize the entire graph with custom colors
    visualize_graph(G)  # Now visualizing the entire graph without limiting nodes
    
     # Plot the metrics
    plot_metrics(precision, recall, f1)

    # Generate Recommendations for All Customers
    all_recommendations = recommend_for_all_customers(G, processed_data, top_n=5)

    # Display Recommendations in a formatted way
    display_recommendations(all_recommendations)
