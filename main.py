from fastapi import FastAPI, Form, HTTPException
import overpy, math
from sentence_transformers import SentenceTransformer
import numpy as np, faiss

app = FastAPI()

# Load embedding model
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Overpass API client
api = overpy.Overpass()

# Global variables
restaurants = []
index = None
vecs = None

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Load restaurants from OpenStreetMap
def load_osm():
    global restaurants, index, vecs

    query = """
    [out:json][timeout:25];
    area["name"="Coventry"]->.a;
    (
      node["amenity"="restaurant"](area.a);
      way["amenity"="restaurant"](area.a);
      relation["amenity"="restaurant"](area.a);
    );
    out center tags;
    """

    result = api.query(query)

    restaurants = []

    # Extract nodes
    for n in result.nodes:
        name = n.tags.get("name", "Unknown")
        cuisine = n.tags.get("cuisine", "")
        lat, lon = float(n.lat), float(n.lon)
        restaurants.append({"name": name, "cuisine": cuisine, "lat": lat, "lon": lon})

    # Extract ways (use center)
    for w in result.ways:
        name = w.tags.get("name", "Unknown")
        cuisine = w.tags.get("cuisine", "")
        lat, lon = float(w.center_lat), float(w.center_lon)
        restaurants.append({"name": name, "cuisine": cuisine, "lat": lat, "lon": lon})

    # Extract relations (use center)
    for r in result.relations:
        name = r.tags.get("name", "Unknown")
        cuisine = r.tags.get("cuisine", "")
        lat, lon = float(r.center_lat), float(r.center_lon)
        restaurants.append({"name": name, "cuisine": cuisine, "lat": lat, "lon": lon})

    # Build embeddings
    texts = [f"{r['name']} {r['cuisine']}" for r in restaurants]
    vecs = MODEL.encode(texts, convert_to_numpy=True)

    # Build FAISS index
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

@app.on_event("startup")
def startup_event():
    load_osm()

@app.post("/recommend")
def recommend(
    lat: float = Form(...),
    lon: float = Form(...),
    cuisine: str = Form(""),
    vibe: str = Form("")
):
    if index is None:
        raise HTTPException(503, "Index not ready")

    # Build query embedding
    qtext = f"{cuisine} {vibe}"
    qvec = MODEL.encode([qtext])

    # Search top 10 matches
    D, I = index.search(qvec, 10)

    results = []
    for i in I[0][:5]:
        r = restaurants[i]
        d = haversine(lat, lon, r["lat"], r["lon"])
        results.append({
            "name": r["name"],
            "cuisine": r["cuisine"],
            "distance_km": round(d, 2)
        })

    return {"results": results}
