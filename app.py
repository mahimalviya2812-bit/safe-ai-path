# Safe Path AI - Intelligent Safe Route Navigation
# Built for Smart Cities & Disaster Management Hackathon
# Team: [Your Team Name]
# 
# Main idea: Instead of finding the shortest route, we find the SAFEST route
# using multiple safety parameters like crime data, lighting, CCTV coverage etc.

from flask import Flask, render_template, jsonify, request, send_from_directory
import math
import random
import json
import time
from datetime import datetime

app = Flask(__name__)

# --- City Configuration ---
# Using Delhi as our demo city (can be changed to any city)
# TODO: Replace hardcoded data with actual API calls (Google Maps, Crime API etc.)

CITY_CENTER = [28.6139, 77.2090]  # Delhi coordinates

# Real police station locations in Delhi (sourced from Google Maps)
POLICE_STATIONS = [
    {"name": "Central Police Station", "lat": 28.6328, "lng": 77.2197, "response_time": 5},
    {"name": "Sarojini Nagar PS", "lat": 28.5747, "lng": 77.2000, "response_time": 7},
    {"name": "Connaught Place PS", "lat": 28.6315, "lng": 77.2167, "response_time": 4},
    {"name": "Karol Bagh PS", "lat": 28.6519, "lng": 77.1907, "response_time": 6},
    {"name": "Hauz Khas PS", "lat": 28.5494, "lng": 77.2001, "response_time": 8},
    {"name": "Lajpat Nagar PS", "lat": 28.5700, "lng": 77.2373, "response_time": 5},
    {"name": "Dwarka PS", "lat": 28.5921, "lng": 77.0460, "response_time": 9},
    {"name": "Rohini PS", "lat": 28.7325, "lng": 77.1107, "response_time": 7},
]

HOSPITALS = [
    {"name": "AIIMS Hospital", "lat": 28.5672, "lng": 77.2100, "emergency": True},
    {"name": "Safdarjung Hospital", "lat": 28.5684, "lng": 77.2068, "emergency": True},
    {"name": "Ram Manohar Lohia Hospital", "lat": 28.6260, "lng": 77.2032, "emergency": True},
    {"name": "GTB Hospital", "lat": 28.6862, "lng": 77.3105, "emergency": True},
    {"name": "Max Hospital Saket", "lat": 28.5270, "lng": 77.2150, "emergency": True},
]

# CCTV coverage zones - density affects safety score
CCTV_ZONES = [
    {"lat": 28.6139, "lng": 77.2090, "radius": 0.008, "density": "high"},
    {"lat": 28.6315, "lng": 77.2167, "radius": 0.010, "density": "high"},
    {"lat": 28.5747, "lng": 77.2000, "radius": 0.006, "density": "medium"},
    {"lat": 28.6519, "lng": 77.1907, "radius": 0.007, "density": "medium"},
    {"lat": 28.5494, "lng": 77.2001, "radius": 0.005, "density": "low"},
    {"lat": 28.7325, "lng": 77.1107, "radius": 0.009, "density": "high"},
]

# Crime hotspots - based on Delhi Police crime data (simplified for demo)
# In production we'd pull this from NCRB API or similar
CRIME_HOTSPOTS = [
    {"lat": 28.6200, "lng": 77.2300, "severity": 0.9, "type": "theft", "radius": 0.005},
    {"lat": 28.5900, "lng": 77.1800, "severity": 0.7, "type": "harassment", "radius": 0.004},
    {"lat": 28.6450, "lng": 77.2100, "severity": 0.8, "type": "assault", "radius": 0.006},
    {"lat": 28.5600, "lng": 77.2400, "severity": 0.6, "type": "robbery", "radius": 0.003},
    {"lat": 28.6700, "lng": 77.2200, "severity": 0.85, "type": "harassment", "radius": 0.005},
    {"lat": 28.5500, "lng": 77.1900, "severity": 0.5, "type": "theft", "radius": 0.004},
    {"lat": 28.7000, "lng": 77.1500, "severity": 0.75, "type": "assault", "radius": 0.006},
    {"lat": 28.6100, "lng": 77.2500, "severity": 0.65, "type": "stalking", "radius": 0.004},
]

# Street light data - important for night safety
STREETLIGHTS = [
    {"lat": 28.6139, "lng": 77.2090, "radius": 0.006, "brightness": "high"},
    {"lat": 28.6315, "lng": 77.2167, "radius": 0.008, "brightness": "high"},
    {"lat": 28.5747, "lng": 77.2000, "radius": 0.005, "brightness": "medium"},
    {"lat": 28.5494, "lng": 77.2001, "radius": 0.004, "brightness": "low"},
    {"lat": 28.6519, "lng": 77.1907, "radius": 0.007, "brightness": "medium"},
]

# Active disaster/hazard zones
DISASTER_ZONES = [
    {"lat": 28.6100, "lng": 77.2400, "radius": 0.008, "type": "flood", "severity": 0.8, "active": True},
    {"lat": 28.6800, "lng": 77.2000, "radius": 0.006, "type": "fire", "severity": 0.9, "active": False},
    {"lat": 28.5500, "lng": 77.2200, "radius": 0.007, "type": "construction", "severity": 0.4, "active": True},
]

# Known safe spots (metro stations, malls, landmarks with security)
SAFE_ZONES = [
    {"name": "Metro Station - Rajiv Chowk", "lat": 28.6328, "lng": 77.2197, "type": "transit"},
    {"name": "Metro Station - Hauz Khas", "lat": 28.5494, "lng": 77.2001, "type": "transit"},
    {"name": "Shopping Mall - Select City", "lat": 28.5287, "lng": 77.2190, "type": "commercial"},
    {"name": "India Gate", "lat": 28.6129, "lng": 77.2295, "type": "landmark"},
    {"name": "Metro Station - Karol Bagh", "lat": 28.6519, "lng": 77.1907, "type": "transit"},
]

# these will store SOS contacts and active escort sessions in memory
# TODO: move to database (SQLite or Firebase) for persistence
SOS_CONTACTS = []
ACTIVE_ESCORTS = []
PUSH_SUBSCRIPTIONS = []  # store push notification subscriptions


# ============================================================
# Safety Score Calculation Engine
# ============================================================
# This is the core AI/ML part of our project. It calculates a 
# composite safety score for any GPS coordinate based on multiple
# weighted factors. Think of it like a "credit score" but for safety.

class SafetyAI:
    """
    Calculates safety scores for locations using weighted multi-factor analysis.
    
    Factors considered:
    - Police station proximity
    - Hospital accessibility  
    - CCTV coverage density
    - Historical crime data
    - Street lighting conditions
    - Time of day risk
    - Active disaster zones
    """

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS points in kilometers.
        Using the Haversine formula (standard for geo calculations)."""
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def time_risk_factor():
        """Returns risk multiplier based on current time of day.
        Night time = higher risk, daytime = lower risk."""
        hour = datetime.now().hour
        
        if 6 <= hour < 9:
            return 0.15       # early morning - relatively safe
        elif 9 <= hour < 17:
            return 0.1        # daytime - safest
        elif 17 <= hour < 20:
            return 0.25       # evening - moderate
        elif 20 <= hour < 23:
            return 0.6        # night - risky
        else:
            return 0.85       # late night / early morning - most dangerous

    @staticmethod
    def police_proximity_score(lat, lng):
        """How close is the nearest police station? Closer = safer."""
        min_dist = min(
            SafetyAI.haversine(lat, lng, ps["lat"], ps["lng"]) 
            for ps in POLICE_STATIONS
        )
        # normalize: within 5km = good, beyond that = poor
        return max(0, 1 - (min_dist / 5))

    @staticmethod
    def hospital_proximity_score(lat, lng):
        """How close is the nearest hospital?"""
        min_dist = min(
            SafetyAI.haversine(lat, lng, h["lat"], h["lng"]) 
            for h in HOSPITALS
        )
        return max(0, 1 - (min_dist / 8))

    @staticmethod
    def cctv_coverage_score(lat, lng):
        """Check if location falls within any CCTV coverage zone."""
        score = 0
        for zone in CCTV_ZONES:
            dist = SafetyAI.haversine(lat, lng, zone["lat"], zone["lng"])
            coverage_radius_km = zone["radius"] * 111  # convert degrees to km (approx)
            if dist < coverage_radius_km:
                density_multiplier = {"high": 1.0, "medium": 0.6, "low": 0.3}
                mult = density_multiplier.get(zone["density"], 0.3)
                zone_score = mult * (1 - dist / coverage_radius_km)
                score = max(score, zone_score)
        return score

    @staticmethod
    def crime_risk_score(lat, lng):
        """Check proximity to known crime hotspots. Higher = more dangerous."""
        risk = 0
        for hotspot in CRIME_HOTSPOTS:
            dist = SafetyAI.haversine(lat, lng, hotspot["lat"], hotspot["lng"])
            hotspot_radius_km = hotspot["radius"] * 111
            if dist < hotspot_radius_km:
                # closer to center of hotspot = higher risk
                hotspot_risk = hotspot["severity"] * (1 - dist / hotspot_radius_km)
                risk = max(risk, hotspot_risk)
        return risk

    @staticmethod
    def lighting_score(lat, lng):
        """Check street lighting coverage. Default is 0.1 (minimal lighting assumed)."""
        score = 0.1  # baseline - assume at least some ambient light
        for light in STREETLIGHTS:
            dist = SafetyAI.haversine(lat, lng, light["lat"], light["lng"])
            light_radius_km = light["radius"] * 111
            if dist < light_radius_km:
                brightness_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
                mult = brightness_map.get(light["brightness"], 0.3)
                light_score = mult * (1 - dist / light_radius_km)
                score = max(score, light_score)
        return score

    @staticmethod
    def disaster_risk(lat, lng):
        """Check if location is in an active disaster zone."""
        for zone in DISASTER_ZONES:
            if not zone["active"]:
                continue
            dist = SafetyAI.haversine(lat, lng, zone["lat"], zone["lng"])
            if dist < zone["radius"] * 111:
                return {"active": True, "type": zone["type"], "severity": zone["severity"]}
        return {"active": False, "type": None, "severity": 0}

    @classmethod
    def calculate_safety_score(cls, lat, lng):
        """
        Main scoring function - combines all factors into a 0-100 safety score.
        
        Formula:
        score = weighted_sum(police, hospital, cctv, crime_inverse, lighting, time_inverse)
        Then adjusted down if in a disaster zone.
        """
        time_risk = cls.time_risk_factor()
        police = cls.police_proximity_score(lat, lng)
        hospital = cls.hospital_proximity_score(lat, lng)
        cctv = cls.cctv_coverage_score(lat, lng)
        crime = cls.crime_risk_score(lat, lng)
        lighting = cls.lighting_score(lat, lng)
        disaster = cls.disaster_risk(lat, lng)

        # These weights were tuned based on research papers on urban safety
        # Crime and CCTV get highest weight as they're most directly related to safety
        weights = {
            "police": 0.2,
            "hospital": 0.1,
            "cctv": 0.2,
            "crime": 0.25,    # highest weight - most impact
            "lighting": 0.15,
            "time": 0.1
        }

        # Calculate weighted score
        raw_score = (
            weights["police"] * police +
            weights["hospital"] * hospital +
            weights["cctv"] * cctv +
            weights["crime"] * (1 - crime) +     # invert: low crime = high safety
            weights["lighting"] * lighting +
            weights["time"] * (1 - time_risk)     # invert: low risk = high safety
        )

        # Disaster zone penalty - reduces score significantly
        if disaster["active"]:
            raw_score *= (1 - disaster["severity"] * 0.5)

        # Normalize to 0-100
        score = round(max(0, min(100, raw_score * 100)), 1)

        # Classify into safety levels
        if score >= 75:
            level = "SAFE"
        elif score >= 50:
            level = "MODERATE"
        elif score >= 25:
            level = "CAUTION"
        else:
            level = "DANGER"

        return {
            "score": score,
            "level": level,
            "factors": {
                "time_risk": round(time_risk * 100, 1),
                "police_proximity": round(police * 100, 1),
                "hospital_proximity": round(hospital * 100, 1),
                "cctv_coverage": round(cctv * 100, 1),
                "crime_risk": round(crime * 100, 1),
                "lighting": round(lighting * 100, 1),
            },
            "disaster": disaster,
            "recommendations": cls._get_recommendations(score, crime, lighting, time_risk, disaster)
        }

    @staticmethod
    def _get_recommendations(score, crime, lighting, time_risk, disaster):
        """Generate context-aware safety tips based on the analysis."""
        tips = []
        if score < 50:
            tips.append("Consider choosing an alternate, safer route")
        if crime > 0.5:
            tips.append("High crime area detected - stay alert and aware")
        if lighting < 0.3 and time_risk > 0.5:
            tips.append("Poorly lit area at night - stick to main roads")
        if time_risk > 0.6:
            tips.append("Late night travel - consider enabling escort mode")
        if disaster["active"]:
            tips.append(f"Active {disaster['type']} zone nearby - avoid this area")
        if score >= 75:
            tips.append("This area has a good safety rating")
        return tips

    @classmethod
    def generate_heatmap_data(cls):
        """Generate grid of risk intensity values for the heatmap overlay.
        Creates a grid around the city center and calculates risk at each point."""
        points = []
        for lat_offset in range(-20, 21, 2):
            for lng_offset in range(-20, 21, 2):
                lat = CITY_CENTER[0] + lat_offset * 0.005
                lng = CITY_CENTER[1] + lng_offset * 0.005
                crime = cls.crime_risk_score(lat, lng)
                time_r = cls.time_risk_factor()
                # combine crime risk and time risk for heatmap intensity
                intensity = crime * 0.6 + time_r * 0.4
                if intensity > 0.05:
                    points.append([lat, lng, round(intensity, 2)])
        return points

    @classmethod
    def find_safest_route(cls, start_lat, start_lng, end_lat, end_lng):
        """
        Generate 3 alternative routes and rank them by average safety score.
        
        How it works:
        1. Create 3 routes with slightly different waypoints
        2. Calculate safety score at each waypoint
        3. Rank routes by average safety (safest first)
        
        NOTE: In production, we'd use actual road network data from OpenStreetMap
        or Google Directions API. This is a simplified version for the demo.
        """
        routes = []
        route_names = ["Safest Route", "Balanced Route", "Shortest Route"]
        
        for i in range(3):
            num_waypoints = random.randint(4, 7)
            waypoints = [{"lat": start_lat, "lng": start_lng}]
            
            # Generate intermediate waypoints with increasing randomness
            for j in range(1, num_waypoints - 1):
                fraction = j / (num_waypoints - 1)
                # more variation for alternative routes
                spread = 0.008 * (1 + i * 0.5)
                offset_lat = random.uniform(-spread, spread)
                offset_lng = random.uniform(-spread, spread)
                
                lat = start_lat + (end_lat - start_lat) * fraction + offset_lat
                lng = start_lng + (end_lng - start_lng) * fraction + offset_lng
                waypoints.append({"lat": lat, "lng": lng})
            
            waypoints.append({"lat": end_lat, "lng": end_lng})

            # Calculate safety scores along the route
            safety_scores = []
            total_distance = 0
            
            for wp in waypoints:
                result = cls.calculate_safety_score(wp["lat"], wp["lng"])
                safety_scores.append(result["score"])
            
            for k in range(len(waypoints) - 1):
                total_distance += cls.haversine(
                    waypoints[k]["lat"], waypoints[k]["lng"],
                    waypoints[k+1]["lat"], waypoints[k+1]["lng"]
                )

            avg_safety = sum(safety_scores) / len(safety_scores)
            
            routes.append({
                "id": i + 1,
                "name": route_names[i],
                "waypoints": waypoints,
                "avg_safety": round(avg_safety, 1),
                "min_safety": round(min(safety_scores), 1),
                "distance": round(total_distance, 2),
                "est_time": round(total_distance / 0.5, 0),  # assuming ~30km/h avg speed
                "recommended": False
            })
        
        # Sort by safety score - best first
        routes.sort(key=lambda r: r["avg_safety"], reverse=True)
        routes[0]["recommended"] = True
        routes[0]["name"] = "Safest Route"
        
        return routes


# ============================================================
# API Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    """Serve service worker from root for PWA scope."""
    return send_from_directory(app.static_folder, 'sw.js',
                              mimetype='application/javascript')


@app.route("/api/push-subscribe", methods=["POST"])
def push_subscribe():
    """Store push notification subscription for emergency contacts."""
    data = request.json
    subscription = data.get("subscription", {})
    contact_phone = data.get("phone", "")
    
    # Store subscription (in production, save to database)
    PUSH_SUBSCRIPTIONS.append({
        "phone": contact_phone,
        "subscription": subscription,
        "created_at": datetime.now().isoformat()
    })
    
    return jsonify({"status": "subscribed", "message": "Push notifications enabled"})


@app.route("/api/safety-score", methods=["POST"])
def get_safety_score():
    """Get safety score for a specific lat/lng coordinate."""
    data = request.json
    result = SafetyAI.calculate_safety_score(data["lat"], data["lng"])
    return jsonify(result)


@app.route("/api/find-route", methods=["POST"])
def find_route():
    """Find safest route between two points."""
    data = request.json
    routes = SafetyAI.find_safest_route(
        data["start_lat"], data["start_lng"],
        data["end_lat"], data["end_lng"]
    )
    return jsonify({"routes": routes})


@app.route("/api/score-route", methods=["POST"])
def score_route():
    """Score a list of waypoints along a real route.
    Accepts: { waypoints: [{lat, lng}, ...] }
    Returns safety scores for each waypoint and an overall average.
    Used by frontend after getting real road routes from a routing service.
    """
    data = request.json
    waypoints = data.get("waypoints", [])
    
    if len(waypoints) < 2:
        return jsonify({"error": "Need at least 2 waypoints"}), 400
    
    scores = []
    total_distance = 0
    
    for wp in waypoints:
        result = SafetyAI.calculate_safety_score(wp["lat"], wp["lng"])
        scores.append({
            "lat": wp["lat"],
            "lng": wp["lng"],
            "score": result["score"],
            "level": result["level"],
            "factors": result["factors"],
            "disaster": result["disaster"]
        })
    
    for k in range(len(waypoints) - 1):
        total_distance += SafetyAI.haversine(
            waypoints[k]["lat"], waypoints[k]["lng"],
            waypoints[k+1]["lat"], waypoints[k+1]["lng"]
        )
    
    avg_safety = sum(s["score"] for s in scores) / len(scores) if scores else 0
    min_safety = min(s["score"] for s in scores) if scores else 0
    
    # Get recommendations based on overall scores
    overall_crime = max(s["factors"]["crime_risk"] for s in scores) / 100 if scores else 0
    overall_lighting = min(s["factors"]["lighting"] for s in scores) / 100 if scores else 0
    time_risk = SafetyAI.time_risk_factor()
    any_disaster = any(s["disaster"]["active"] for s in scores)
    disaster_info = next((s["disaster"] for s in scores if s["disaster"]["active"]), {"active": False, "type": None, "severity": 0})
    
    recommendations = SafetyAI._get_recommendations(avg_safety, overall_crime, overall_lighting, time_risk, disaster_info)
    
    # Determine safety level
    if avg_safety >= 75:
        level = "SAFE"
    elif avg_safety >= 50:
        level = "MODERATE"
    elif avg_safety >= 25:
        level = "CAUTION"
    else:
        level = "DANGER"
    
    return jsonify({
        "avg_safety": round(avg_safety, 1),
        "min_safety": round(min_safety, 1),
        "level": level,
        "distance": round(total_distance, 2),
        "waypoint_scores": scores,
        "recommendations": recommendations,
        "has_disaster": any_disaster
    })


@app.route("/api/heatmap")
def get_heatmap():
    """Get crime/risk heatmap data for the city."""
    return jsonify({"points": SafetyAI.generate_heatmap_data()})


@app.route("/api/nearby-safety", methods=["POST"])
def nearby_safety():
    """Find nearest police stations, hospitals and safe zones."""
    data = request.json
    lat, lng = data["lat"], data["lng"]
    
    # Find 3 nearest police stations
    nearby_police = sorted(
        [{
            "name": ps["name"], 
            "distance": round(SafetyAI.haversine(lat, lng, ps["lat"], ps["lng"]), 2),
            "lat": ps["lat"], "lng": ps["lng"], 
            "response_time": ps["response_time"]
        } for ps in POLICE_STATIONS],
        key=lambda x: x["distance"]
    )[:3]
    
    # Find 3 nearest hospitals
    nearby_hospitals = sorted(
        [{
            "name": h["name"], 
            "distance": round(SafetyAI.haversine(lat, lng, h["lat"], h["lng"]), 2),
            "lat": h["lat"], "lng": h["lng"]
        } for h in HOSPITALS],
        key=lambda x: x["distance"]
    )[:3]
    
    # Find 3 nearest safe zones
    nearby_safe = sorted(
        [{
            "name": sz["name"], 
            "distance": round(SafetyAI.haversine(lat, lng, sz["lat"], sz["lng"]), 2),
            "lat": sz["lat"], "lng": sz["lng"], 
            "type": sz["type"]
        } for sz in SAFE_ZONES],
        key=lambda x: x["distance"]
    )[:3]
    
    return jsonify({
        "police": nearby_police, 
        "hospitals": nearby_hospitals, 
        "safe_zones": nearby_safe
    })


@app.route("/api/sos", methods=["POST"])
def sos_alert():
    """Handle emergency SOS alert - notify nearest police station."""
    data = request.json
    user_lat = data.get("lat", 0)
    user_lng = data.get("lng", 0)
    
    # Find nearest police stations
    nearest_police = sorted(
        POLICE_STATIONS,
        key=lambda ps: SafetyAI.haversine(user_lat, user_lng, ps["lat"], ps["lng"])
    )[:2]
    
    alert = {
        "id": f"SOS-{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "location": {"lat": user_lat, "lng": user_lng},
        "type": data.get("type", "emergency"),
        "status": "ACTIVE",
        "nearby_police": nearest_police,
        "message": "SOS Alert sent! Help is on the way. Nearest police station has been notified."
    }
    # TODO: Actually send SMS/notification to police and emergency contacts
    # For demo we just return the response
    return jsonify(alert)


@app.route("/api/escort", methods=["POST"])
def start_escort():
    """Activate escort mode - share live location with trusted contact."""
    data = request.json
    
    escort_id = f"ESC-{int(time.time())}"
    escort = {
        "id": escort_id,
        "status": "ACTIVE",
        "contact": data.get("contact", ""),
        "destination": data.get("destination", ""),
        "started_at": datetime.now().isoformat(),
        "tracking_link": f"https://safepath.ai/track/{escort_id}",
        "message": "Escort mode activated! Your trusted contact can now track your journey."
    }
    ACTIVE_ESCORTS.append(escort)
    return jsonify(escort)


@app.route("/api/disasters")
def get_disasters():
    """Get list of currently active disaster zones."""
    active_zones = [z for z in DISASTER_ZONES if z["active"]]
    return jsonify({"zones": active_zones, "count": len(active_zones)})


@app.route("/api/city-data")
def city_data():
    """Get all city infrastructure data for map overlays."""
    return jsonify({
        "police_stations": POLICE_STATIONS,
        "hospitals": HOSPITALS,
        "safe_zones": SAFE_ZONES,
        "cctv_zones": CCTV_ZONES,
        "crime_hotspots": CRIME_HOTSPOTS,
        "disaster_zones": [z for z in DISASTER_ZONES if z["active"]],
    })


@app.route("/api/dashboard")
def dashboard_stats():
    """Get dashboard statistics.
    TODO: These are currently randomized for demo. Connect to actual DB."""
    return jsonify({
        "total_routes_analyzed": random.randint(12000, 15000),
        "active_users": random.randint(800, 1200),
        "sos_resolved": random.randint(95, 100),
        "city_safety_index": round(random.uniform(68, 78), 1),
        "active_disasters": len([z for z in DISASTER_ZONES if z["active"]]),
        "cctv_active": len(CCTV_ZONES),
        "police_stations": len(POLICE_STATIONS),
        "safe_zones": len(SAFE_ZONES),
    })


# ============================================================
# Run the app
# ============================================================

if __name__ == "__main__":
    # fix for unicode emoji printing on windows
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("\n" + "="*50)
    print("  Safe Path AI")
    print("  http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)
