import osmnx as ox
import geopandas as gpd
import networkx as nx
import json
from progress.bar import Bar

def load_graph_from_file(file_path):
    if file_path.endswith(".osm") or file_path.endswith(".xml"):
        return ox.graph_from_xml(file_path)
    else:
        raise ValueError("Unsupported file format. Use .osm or .xml")

def extract_node_data(graph: nx.MultiDiGraph):
    nodes = {}
    for node_id, data in graph.nodes(data=True):
        nodes[node_id] = {
            "id": str(node_id),
            "lat": data["y"],
            "lng": data["x"],
            "name": f"Intersection at ({round(data['y'], 5)}, {round(data['x'], 5)})",
            "connected_roads": set()
        }
    return nodes

def extract_road_data(graph: nx.MultiDiGraph, nodes: dict):
    roads = []
    road_name_by_id = {}


    bar = Bar("Extracting data...",max=len(graph.edges))

    for u, v, key, data in graph.edges(keys=True, data=True):

        road_id = f"{u}-{v}-{key}"
        road_name = data.get("name", "Unnamed Road")
        road_type = data.get("highway", "road")
        length = data.get("length", 0)

        
        if road_name == "Unnamed Road":
            bar.next()
            continue  # skip this edge

        # Track connections
        if isinstance(road_name, list):
            for name in road_name:
                nodes[u]["connected_roads"].add(name)
        else:
            nodes[u]["connected_roads"].add(road_name)
        if isinstance(road_name, list):
            for name in road_name:
                nodes[v]["connected_roads"].add(name)
        else:
            nodes[v]["connected_roads"].add(road_name)

        roads.append({
            "id": road_id,
            "name": road_name,
            "from_node": str(u),
            "to_node": str(v),
            "length": length,
            "type": road_type
        })

        road_name_by_id[road_id] = road_name
        bar.next()
    bar.finish()

    return roads, road_name_by_id

def convert_to_rag_chunks(roads, nodes, road_name_by_id):
    node_map = {n["id"]: n for n in nodes.values()}
    chunks = []

    bar = Bar("Converting to chunks...",max=len(roads))

    for road in roads:
        from_node = node_map.get(road["from_node"])
        to_node = node_map.get(road["to_node"])

        road_names = road["name"]
        if isinstance(road_names, list):
            road_names = set(road_names)
        else:
            road_names = {road_names}

        from_roads = from_node["connected_roads"] - road_names
        to_roads = to_node["connected_roads"] - road_names
        connected = list(from_roads.union(to_roads))

        chunk_text = (
            f"{road['name']} runs from {from_node['name']} to {to_node['name']}. "
            f"It connects to roads like {', '.join(connected[:5])}."
        )

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "road_name": road["name"],
                "from": from_node["name"],
                "to": to_node["name"],
                "type": road["type"],
                "connected_roads": connected
            }
        })

        bar.next()
    bar.finish()

    return chunks

def save_chunks(chunks, out_file: str):
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    input_file = "./data/main_roads.osm"
    print(f"Loading road network from {input_file}...")
    graph = load_graph_from_file(input_file)
    nodes = extract_node_data(graph)
    roads, road_name_map = extract_road_data(graph, nodes)
    chunks = convert_to_rag_chunks(roads, nodes, road_name_map)
    save_chunks(chunks, "rag_road_chunks.json")
    print(f"Saved {len(chunks)} RAG chunks to rag_road_chunks.json")
