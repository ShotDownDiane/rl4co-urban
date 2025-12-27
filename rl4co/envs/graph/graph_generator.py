import numpy as np
import networkx as nx
from scipy.spatial import Delaunay, distance_matrix
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

class SpatialGraphGenerator:
    """
    Given a set of 2D locations, generate a connected graph structure.
    """
    
    @staticmethod
    def generate_delaunay(locs: np.ndarray) -> nx.Graph:
        """
        Method 1: Delaunay Triangulation (The most "road-like" planar graph).
        Guarantees connectivity and planarity (no edge crossings).
        """
        n_points = len(locs)
        tri = Delaunay(locs)
        G = nx.Graph()
        G.add_nodes_from(range(n_points))
        
        # Add edges from simplices (triangles)
        for simplex in tri.simplices:
            # simplex is [p1, p2, p3]
            G.add_edge(simplex[0], simplex[1], weight=np.linalg.norm(locs[simplex[0]] - locs[simplex[1]]))
            G.add_edge(simplex[1], simplex[2], weight=np.linalg.norm(locs[simplex[1]] - locs[simplex[2]]))
            G.add_edge(simplex[2], simplex[0], weight=np.linalg.norm(locs[simplex[2]] - locs[simplex[0]]))
            
        # Store pos for plotting
        for i, coord in enumerate(locs):
            G.nodes[i]['pos'] = coord
            
        return G

    @staticmethod
    def generate_knn_mst(locs: np.ndarray, k: int = 3) -> nx.Graph:
        """
        Method 2: KNN + MST (The "Robust" method).
        KNN creates local clusters, MST acts as the backbone to guarantee global connectivity.
        """
        n_points = len(locs)
        dist_matrix = distance_matrix(locs, locs)
        G = nx.Graph()
        G.add_nodes_from(range(n_points))
        
        # 1. Add KNN edges (Local density)
        # mode='distance' returns actual distances as weights
        knn_adj = kneighbors_graph(locs, k, mode='distance', include_self=False)
        knn_graph = nx.from_scipy_sparse_array(knn_adj)
        G.add_edges_from(knn_graph.edges(data=True))
        
        # 2. Add MST edges (Global connectivity)
        # Create a full graph first to compute MST
        full_G = nx.complete_graph(n_points)
        for u, v in full_G.edges():
            full_G[u][v]['weight'] = dist_matrix[u][v]
        
        mst = nx.minimum_spanning_tree(full_G)
        G.add_edges_from(mst.edges(data=True))
        
        # Store pos
        for i, coord in enumerate(locs):
            G.nodes[i]['pos'] = coord
            
        return G

    @staticmethod
    def generate_radius_mst(locs: np.ndarray, radius: float = 0.2) -> nx.Graph:
        """
        Method 3: Radius Graph + MST.
        Connects all nodes within distance R. Adds MST to fix disconnected components.
        """
        n_points = len(locs)
        dist_matrix = distance_matrix(locs, locs)
        G = nx.Graph()
        
        # 1. Radius connections
        rows, cols = np.where((dist_matrix <= radius) & (dist_matrix > 0))
        for u, v in zip(rows, cols):
            if u < v: # Avoid duplicates
                G.add_edge(u, v, weight=dist_matrix[u][v])
                
        # 2. Add MST to ensure connectivity (Fix islands)
        full_G = nx.complete_graph(n_points)
        for u, v in full_G.edges():
            full_G[u][v]['weight'] = dist_matrix[u][v]
        mst = nx.minimum_spanning_tree(full_G)
        G.add_edges_from(mst.edges(data=True))
        
        for i, coord in enumerate(locs):
            G.nodes[i]['pos'] = coord
            
        return G

# --- Visualization Helper ---
def plot_graph(G, title):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos, node_size=30, node_color='blue', edge_color='gray', with_labels=False, alpha=0.6)
    plt.title(f"{title}\nEdges: {G.number_of_edges()}")
    plt.savefig(f"/root/autodl-tmp/rl4co-urban/results/figs/{title}.png")
    plt.close()

# --- Example Usage ---
if __name__ == "__main__":
    # Generate random locations
    np.random.seed(42)
    num_nodes = 50
    locs = np.random.rand(num_nodes, 2)

    # 1. Delaunay (Planar, Road-like)
    G_delaunay = SpatialGraphGenerator.generate_delaunay(locs)
    plot_graph(G_delaunay, "Delaunay Triangulation (Guaranteed Connected)")

    # 2. KNN + MST (Controlled Sparsity)
    G_knn = SpatialGraphGenerator.generate_knn_mst(locs, k=3)
    plot_graph(G_knn, "KNN (k=3) + MST Union")

    # 3. Radius + MST (Controlled Sparsity)
    G_radius = SpatialGraphGenerator.generate_radius_mst(locs, radius=0.2)
    plot_graph(G_radius, "Radius (R=0.2) + MST Union")