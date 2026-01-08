import requests

def get_precise_location():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        if data['status'] == 'success':
            return f"{data['lat']},{data['lon']}"
        return "Unknown"
    except Exception as e:
        return "Unknown"
